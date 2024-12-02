from flask import Flask, render_template, request, send_file
import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Define Model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load model
model_path = './outputs/model.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN()

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)  # Load model weights properly
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.to(device)
model.eval()

# Preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# De-processing the image after deblurring
def deprocess_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deblur', methods=['POST'])
def deblur():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return "Invalid file format.", 400
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image).squeeze(0)
        
        output_image = deprocess_image(output)
        buffer = io.BytesIO()
        output_image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/jpeg', as_attachment=True, download_name="deblurred_image.jpg")
    except Exception as e:
        return f"Error processing image: {e}", 500

if __name__ == '__main__':
    os.makedirs('./outputs', exist_ok=True)
    app.run(debug=True)
