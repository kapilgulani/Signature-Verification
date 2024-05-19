import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust the size according to your input dimensions
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
# Define the Siamese Network architecture (as previously defined in your training script)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = CNN()
        self.final = nn.Linear(128, 1)  # Output layer to predict the similarity

    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        l1_distance = torch.abs(output1 - output2)
        similarity_score = self.final(l1_distance)
        return torch.sigmoid(similarity_score)


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image


def predict(img_path1, img_path2, model, device, transform):
    # Load and transform images
    img1 = load_image(img_path1, transform).unsqueeze(0).to(device)  # Add batch dimension
    img2 = load_image(img_path2, transform).unsqueeze(0).to(device)  # Add batch dimension

    # Set model to evaluation mode and make prediction
    model.eval()
    with torch.no_grad():
        output = model(img1, img2).item()  # Get the raw probability from the model output
        predicted = output > 0.90  # Assuming threshold of 0.5

    return "Genuine" if predicted else "Forged", output * 100  # Return percentage


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_image1> <path_to_image2>")
        sys.exit(1)

    img_path1 = sys.argv[1]
    img_path2 = sys.argv[2]

    # Transformations must match those used during training
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the model
    model = SiameseNetwork()  # Ensure the network architecture is correctly defined
    model_path = 'best_model.pth'  # Path to the model's state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Make prediction
    result, similarity = predict(img_path1, img_path2, model, device, transformations)
    print(f"The images are {result}. Similarity: {similarity:.2f}%.")


if __name__ == "__main__":
    main()