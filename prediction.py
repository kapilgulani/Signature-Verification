import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # Load the image file
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    transform = transforms.Compose([
        transforms.Resize((105, 105)),  # Resize the image to 105x105
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])
    return transform(img)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def distance_to_similarity(distance, average_distance=20):
    similarity = max(0, (average_distance - distance) / average_distance) * 100
    return similarity

# Load and preprocess images
image1 = load_and_preprocess_image('../kg1.jpg').unsqueeze(0).to(device)
image2 = load_and_preprocess_image('../kg2.jpg').unsqueeze(0).to(device)

# Compute outputs from the model
output1, output2 = model(image1, image2)

# Calculate the Euclidean distance
euclidean_distance = F.pairwise_distance(output1, output2).item()

# Convert distance to similarity percentage
similarity_percentage = distance_to_similarity(euclidean_distance)

# Print results
print(f"Similarity: {similarity_percentage:.2f}%")
print("Classification: Forged" if similarity_percentage < 80 else "Classification: Genuine")

def show_image(image, text=None, should_save=False):
    image = image.squeeze(0).detach().cpu()
    np_image = image.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(np_image, (1, 2, 0)), cmap='gray')
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    if should_save:
        plt.savefig('output_image.png')
    plt.show()

# Display images
show_image(image1, "Image 1")
show_image(image2, "Image 2")
