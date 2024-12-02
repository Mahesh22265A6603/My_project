import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
import models
from torchvision import transforms

# Ensure output directory exists
os.makedirs('outputs/test_deblurred_images', exist_ok=True)

def save_decoded_image(img_tensor, filepath):
    """
    Saves the output image ensuring the correct color range and dimensions.
    """
    img_tensor = img_tensor.squeeze(0).clamp(0, 1)  # Clamp values to [0, 1]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert to BGR for saving
    print(f"Saved image: {filepath}")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the trained model
model = models.CNN().to(device).eval()
model_path = 'outputs/best_deblur_model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'. Please ensure the file exists.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define transforms with normalization for input consistency
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Ensure uniform size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean & std
])

# Inverse normalization for saving output images
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
])

def preprocess_image(image_path):
    """
    Reads, converts, and preprocesses an image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {image_path}, shape: {image.shape}")
    return transform(image).unsqueeze(0)

# Specify the input image name
image_name = 'image_1'
image_path = f"test_data/gaussian_blurred/{image_name}.jpg"

try:
    # Preprocess and display input image
    image_tensor = preprocess_image(image_path).to(device)
    print(f"Input image tensor shape: {image_tensor.shape}")

    # Inference
    with torch.no_grad():
        output_tensor = model(image_tensor)
        output_tensor = inverse_transform(output_tensor.squeeze(0)).unsqueeze(0)  # Undo normalization for saving
        deblurred_image_path = f"outputs/test_deblurred_images/deblurred_image_{image_name}.jpg"
        save_decoded_image(output_tensor.cpu(), deblurred_image_path)

    # Save original resized blurred image for comparison
    original_save_path = f"outputs/test_deblurred_images/original_blurred_image_{image_name}.jpg"
    original_resized = cv2.resize(cv2.imread(image_path), (224, 224))
    cv2.imwrite(original_save_path, original_resized)
    print(f"Original resized image saved at {original_save_path}.")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
