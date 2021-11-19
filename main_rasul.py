import os
import random
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import fnmatch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

#Turn a link to a directory into an object containing manipulatable references
class MusicDataset(Dataset):
    def __init__(self, my_dir):
        self.dir = my_dir
        self.files = sorted(self.find_files(my_dir))

    #number of file
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]
    def find_files(self, directory, pattern='*.png'):
    # Recursively finds all files matching the pattern (from lab 1.3)
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

def main(device, datapath):
    dataset = MusicDataset(m_dir)
    print(len(dataset))

if __name__ == "__main__":
    # Define the device and data repository
    device = 'cpu'
    m_dir = '../music_data/images_original/'
    main(device, m_dir)

