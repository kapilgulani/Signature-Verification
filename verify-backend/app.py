from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
from dotenv import load_dotenv
import os
import requests
import base64
import os
from PIL import Image
import numpy as np
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage import filters, transform, img_as_ubyte, io, color, img_as_ubyte
from skimage.io import imread
from typing import Tuple
from collections import OrderedDict


load_dotenv()

app = Flask(__name__)
CORS(app)

def crop_center(img: np.ndarray,
                size: Tuple[int, int]) -> np.ndarray:
    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = img[start_y: start_y + size[0], start_x:start_x + size[1]]
    return cropped

def resize_image(img: np.ndarray,
                 size: Tuple[int, int]) -> np.ndarray:
    height, width = size

    # Check which dimension needs to be cropped
    # (assuming the new height-width ratio may not match the original size)
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    # Resize the image (will still be larger than new_size in one dimension)
    img = transform.resize(img, (resize_height, resize_width),
                           mode='constant', anti_aliasing=True, preserve_range=True)

    img = img.astype(np.uint8)
    # Crop to exactly the desired new_size, using the middle of the image:
    if width_ratio > height_ratio:
        start = int(round((resize_width-width)/2.0))
        return img[:, start:start + width]
    else:
        start = int(round((resize_height-height)/2.0))
        return img[start:start + height, :]

# Define the preprocessing functions directly
def preprocess_signature(img: np.ndarray, canvas_size: (int, int), img_size: (int, int) = (170, 242), input_size: (int, int) = (150, 220)) -> np.ndarray:
    """Preprocess signature for prediction."""
    img = img.astype(np.uint8)
    centered = normalize_image(img, canvas_size)
    inverted = 255 - centered
    resized = resize_image(inverted, img_size)
    cropped = crop_center(resized, input_size) if input_size is not None and input_size != img_size else resized
    return cropped

def normalize_image(img: np.ndarray, canvas_size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    """Normalize images within a canvas."""
    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)
    threshold = filters.threshold_otsu(img)
    binarized_image = blurred_image > threshold
    r, c = np.where(binarized_image == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())
    cropped = img[r.min(): r.max(), c.min(): c.max()]
    img_rows, img_cols = cropped.shape
    max_rows, max_cols = canvas_size
    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center
    if img_rows > max_rows or img_cols > max_cols:
        r_start, c_start = 0, 0  # Adjust start to top-left corner if cropping is needed
    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    normalized_image[r_start:r_start + img_rows, c_start:c_start + img_cols] = cropped
    return normalized_image

# Define the Siamese Network within the Flask script
class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_mish(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_mish(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_mish(256, 384, 3, pad=1)),
            ('conv4', conv_bn_mish(384, 384, 3, pad=1)),
            ('conv5', conv_bn_mish(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_mish(256 * 3 * 5, 2048)),
            ('fc2', linear_bn_mish(self.feature_space_size, self.feature_space_size)),
        ]))
    
    def forward_once(self, img):
        x = self.conv_layers(img)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc_layers(x)
        return x
        
    def forward(self, img1, img2):
        
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)
        # forward pass of input 1
        output1 = self.forward_once(img1)
        # forward pass of input 2
        output2 = self.forward_once(img2)
        return output1, output2

def conv_bn_mish(in_channels, out_channels, kernel_size,  stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('mish', nn.Mish()),
    ]))


def linear_bn_mish(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('mish', nn.Mish()),
    ]))

class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()

        self.model = SigNet()
        state_dict, _, _ = torch.load("signet.pth")
        self.model.load_state_dict(state_dict)
        
        self.probs = nn.Linear(4, 1)
        self.projection2d = nn.Linear(self.model.feature_space_size, 2)
    
    def forward_once(self, img):
        x = self.model.forward_once(img)
        return x
        
    def forward(self, img1, img2):
        
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        # forward pass of input 1
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)
        embedding1 = self.forward_once(img1)
        # forward pass of input 2
        embedding2 = self.forward_once(img2)
        
            #print("Project embeddings into 2d space")
        embedding1 = self.projection2d(embedding1)
        embedding2 = self.projection2d(embedding2)
            # Classification
        output = torch.cat([embedding1, embedding2], dim=1)
        output= self.probs(output)
        
        return embedding1, embedding2, output

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseModel().to(device)
# model.load_state_dict(torch.load("best_model_21.pt", map_location=device)['model'])
model.eval()

def decode_image(data_uri):
    """Decode a base64-encoded image to a numpy array."""
    header, encoded = data_uri.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image = io.imread(BytesIO(binary_data), as_gray=True)
    return img_as_ubyte(image)

@app.route('/create_user', methods=['POST'])
@cross_origin()
def create_user():
    # Parse JSON data sent from a client
    data = request.get_json()

    # Extract fields
    name = data.get('name')
    email = data.get('email')
    genuine_signature = data.get('genuineSignature')  # Base64-encoded image

    api_url = "https://us-west-2.aws.neurelo.com/rest/user_details/__one"
    headers = {
        "X-API-KEY": os.getenv('NEURELO_VALUE'),  # Get API key from environment variable
        "Content-Type": "application/json"
    }

    api_payload = {
        "name": name,
        "email": email,
        "signature_image": genuine_signature
    }

    response = requests.post(api_url, json=api_payload, headers=headers)

    # Check if the request to the external API was successful 
    if response.status_code == 201:
        # Parse the JSON response from the external API
        api_data = response.json()
        return jsonify({"status": "success", "message": "User created successfully!"})
    else:
        print(f"Failed to send data to external API. \n Status code: {response.status_code}")
        return jsonify({
            "status": "error",
            "message": "Failed to send data to external API.",
        })

@app.route('/get_users', methods=['GET'])
@cross_origin()
def get_users():
    api_url = "https://us-west-2.aws.neurelo.com/rest/user_details"
    
    headers = {
        "X-API-KEY": os.getenv('NEURELO_VALUE'),
        "Content-Type": "application/json"
    }

    response = requests.get(api_url, headers=headers)

    return response.json()

@app.route('/verify_signature', methods=['POST'])
@cross_origin()
def verify_signature():
    data = request.get_json()
    if not data or 'image1' not in data or 'image2' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    try:
        img1_array = decode_image(data['image1'])
        img2_array = decode_image(data['image2'])

        img1_processed = preprocess_signature(img1_array, (840, 1360), (256, 256))
        img2_processed = preprocess_signature(img2_array, (840, 1360), (256, 256))

        img1_tensor = torch.tensor(img1_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img2_tensor = torch.tensor(img2_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Ensure model is in evaluation mode and on the correct device
        model.eval()
        device = torch.device('cpu')  # Assuming you are using CPU for Flask
        model.to(device)

        # Inference
        with torch.no_grad():
            op1, op2, confidence = model(img1_tensor, img2_tensor)
            confidence = torch.sigmoid(confidence).item()  # Convert output to probability
            cos_sim = F.cosine_similarity(op1, op2).item()  # Calculate cosine similarity

        return jsonify({
            'similarity_percentage': f"{cos_sim * 100:.2f}%",
            'classification': 'Forged' if cos_sim < 0.7 else 'Genuine',
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Hello, testing..."

if __name__ == '__main__':
    app.run(debug=True)