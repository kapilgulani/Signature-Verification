from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm  # for progress bars
import torch.nn as nn
import pandas as pd

# Load the dataset
file_path = 'saved_data.csv'
data = pd.read_csv(file_path)
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

            nn.Linear(128, 2))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
# Assuming the model and device are already set up
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert('L')
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

def compute_distance(row):
    img1 = load_image(f'../sign_data/full/{row["image1"]}')  # Adjust path as necessary
    img2 = load_image(f'../sign_data/full/{row["image2"]}')
    img1, img2 = img1.to(device), img2.to(device)
    with torch.no_grad():
        output1, output2 = model(img1, img2)
    return F.pairwise_distance(output1, output2).item()

# Compute distances for all pairs
distances = []
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    distances.append(compute_distance(row))

# Analyzing distances
avg_distance = sum(distances) / len(distances)

print("Avg Distance is ", avg_distance)

def give_avg_distance():
    return avg_distance
