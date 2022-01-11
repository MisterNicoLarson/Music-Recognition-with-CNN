import os
import random
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import fnmatch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
#Turn a link to a directory into an object containing manipulatable references



class MusicDataset(Dataset):
    def __init__(self, my_dir, train=False, dev=False, test=False): 
        self.dir = my_dir
        self.files = self.find_files(my_dir)
        self.labels = self.create_labels(self.files)
        self.classes = self.create_codes()
        self.test = self.files[4::5]
        self.dev = self.files[3::5]
        self.train = [f for f in self.files if not f in (self.test or self.dev)]
        self.train_labels = self.create_labels(self.train)
        self.train = zip(self.train, self.train_labels)
        self.dev_labels = self.create_labels(self.dev)
        self.dev = zip(self.dev, self.dev_labels)
        self.test_labels = self.create_labels(self.test)
        self.test = zip(self.test, self.test_labels)
        if train:
            self.files = list(self.train)
        elif dev:
            self.files = list(self.dev)
        else:
            self.files = list(self.test)
    #number of files
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im = io.imread(self.files[index][0])
        #data label pair where the label is an integer standing for the genre
        return torch.tensor(im, dtype=float), self.classes[self.files[index][1]]
    def __str__(self):
        return str(self.files)
    def get_labels(self):
        return self.labels
    #create int codes for each genre
    def create_labels(self, portion):
        #use folder names as labels
        return [f.split("/")[1] for f in portion]
    def create_codes(self):
        categories = list(set(self.labels))
        return {l : i for i,l in enumerate(categories)}

    def find_files(self, directory, pattern='*.png'):
     #Recursively finds all files matching the pattern (from lab 1.3)
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return sorted(files)



if __name__ == "__main__":
    # Define the device and data repository
    device = 'cpu'
    datapath = 'images_original/'
    train = MusicDataset(datapath, train=True)
    test = MusicDataset(datapath, test=True)
    batch_size = 8
    train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)
    print(len(train_dataloader))
    print(len(test_dataloader))
   # main(device, m_dir)

