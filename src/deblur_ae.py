import os
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations import Normalize, Resize, Compose
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
import models  # Ensure models.py contains your CNN model
import torch.nn.functional as F

# Argument parser for dynamic configurations
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Learning rate')
args = vars(parser.parse_args())

# Device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Directories for inputs and outputs
os.makedirs('outputs/saved_images', exist_ok=True)

# Normalization and Denormalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def denormalize(img):
    img = img * torch.tensor(std).view(1, 3, 1, 1).to(img.device) + torch.tensor(mean).view(1, 3, 1, 1).to(img.device)
    return img.clamp(0, 1)

def save_decoded_image(img, name):
    img = denormalize(img)
    save_image(img, name)

# Improved Wiener Filter for Gaussian Deblurring
def wiener_filter(img, kernel_size=7, sigma=1.5, noise_var=0.05):
    channels = cv2.split(img)
    filtered_channels = []
    for channel in channels:
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        kernel_ft = np.fft.fft2(kernel, s=channel.shape)
        img_ft = np.fft.fft2(channel)
        kernel_ft_conj = np.conj(kernel_ft)
        wiener_ft = kernel_ft_conj / (np.abs(kernel_ft) ** 2 + noise_var) * img_ft
        deblurred_channel = np.abs(np.fft.ifft2(wiener_ft))
        filtered_channels.append(np.uint8(deblurred_channel))
    deblurred_img = cv2.merge(filtered_channels)
    return deblurred_img

# Load and preprocess data
gauss_blur = sorted(os.listdir('input/gaussian_blurred/'))
sharp = sorted(os.listdir('input/sharp/'))

# Ensure the number of blurry and sharp images match
assert len(gauss_blur) == len(sharp), "Number of blurred and sharp images don't match!"

# Splitting the dataset into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(gauss_blur, sharp, test_size=0.25)

transform = Compose([
    Resize(224, 224),
    Normalize(mean=mean, std=std),
    ToTensorV2(),
])

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, apply_wiener=True):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transform
        self.apply_wiener = apply_wiener

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        blur_image = cv2.cvtColor(cv2.imread(f"input/gaussian_blurred/{self.X[idx]}"), cv2.COLOR_BGR2RGB)
        if self.apply_wiener:
            blur_image = wiener_filter(blur_image)
        blur_image = self.transforms(image=blur_image)["image"]
        if self.y:
            sharp_image = cv2.cvtColor(cv2.imread(f"input/sharp/{self.y[idx]}"), cv2.COLOR_BGR2RGB)
            sharp_image = self.transforms(image=sharp_image)["image"]
            return blur_image, sharp_image
        return blur_image

train_data = DeblurDataset(x_train, y_train, apply_wiener=True)
val_data = DeblurDataset(x_val, y_val, apply_wiener=False)
trainloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
valloader = DataLoader(val_data, batch_size=args['batch_size'], shuffle=False)

# Perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights='VGG16_Weights.IMAGENET1K_V1').features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        return self.mse(output_features, target_features)

model = models.CNN().to(device)
mse_loss = nn.MSELoss()
perceptual_loss = PerceptualLoss()
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

def train_epoch(model, loader, optimizer, mse_loss, perceptual_loss):
    model.train()
    total_loss = 0.0
    for blur, sharp in tqdm(loader, desc="Training"):
        blur, sharp = blur.to(device), sharp.to(device)
        optimizer.zero_grad()
        output = model(blur)
        loss = mse_loss(output, sharp) + 0.2 * perceptual_loss(output, sharp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, mse_loss, perceptual_loss):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for blur, sharp in tqdm(loader, desc="Validation"):
            blur, sharp = blur.to(device), sharp.to(device)
            output = model(blur)
            loss = mse_loss(output, sharp) + 0.2 * perceptual_loss(output, sharp)
            total_loss += loss.item()
    return total_loss / len(loader)

# Training and Validation Loop
best_val_loss = float('inf')
for epoch in range(args['epochs']):
    print(f"\nEpoch {epoch+1}/{args['epochs']}")
    
    train_loss = train_epoch(model, trainloader, optimizer, mse_loss, perceptual_loss)
    print(f"Training Loss: {train_loss:.6f}")

    val_loss = validate(model, valloader, mse_loss, perceptual_loss)
    print(f"Validation Loss: {val_loss:.6f}")

    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'outputs/best_deblur_model.pth')
        print("Model saved!")

    # Optionally, save some decoded images from the validation set
    if (epoch + 1) % 5 == 0:  # Save every 5 epochs for example
        model.eval()
        with torch.no_grad():
            for i, (blur, sharp) in enumerate(valloader):
                blur = blur.to(device)
                output = model(blur)
                save_decoded_image(output[0], f"outputs/saved_images/deblurred_epoch{epoch+1}_sample{i}.png")
                if i >= 5:  # Save only first 5 samples
                    break

print("Training complete!")
