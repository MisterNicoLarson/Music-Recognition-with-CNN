import os
import random
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import DataLoader, random_split, Subset
from musicdata import MusicDataset


class MusicCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act_fn):
        super(CNNClassif, self).__init__()




# Choose one (or several) transform(s) to preprocess the data
data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

# Define the device and data repository
device = 'cpu'
datapath = 'images_original/'
train_data = MusicDataset(datapath, train=True)
test_data = MusicDataset(datapath, test=True)
batch_size = 8
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

input_size = train_data[0][0].shape
print(input_size)
