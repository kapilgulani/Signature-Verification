import pandas as pd
from PIL import Image
import os
import numpy as np
import torch

class Create_Dataset():

    def __init__(self, train_csv=None, train_directory=None, transform=None):
        # used to prepare the labels and images path
        self.train_dataframe = pd.read_csv(train_csv)
        self.train_dataframe.columns = ["image1", "image2", "label"]
        self.train_directory = train_directory
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.train_directory, self.train_dataframe.iat[index, 0])
        image2_path = os.path.join(self.train_directory, self.train_dataframe.iat[index, 1])
        # print(image1_path,image2_path)

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.train_dataframe.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.train_dataframe)