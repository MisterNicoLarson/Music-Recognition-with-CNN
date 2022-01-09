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


#copied from lab3
class MusicCNN(nn.Module):
    def (self, num_channels1=16, num_channels2=32, num_classes=10):
        super(CNNClassif_bnorm, self).__init__()
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2))
            
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2))
        
        self.lin_layer = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        
        out = self.cnn_layer1(x)
        out = self.cnn_layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.lin_layer(out)
        return out 
    
 # Training function
def training_classifier(model, train_dataloader, num_epochs, loss_fn, learning_rate, device='cpu', verbose=True):

    # Set the model
    model = model.to(device)
    model.train()
    
    # define the optimizer (SGD)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Initialize a list to save the training loss over epochs
    loss_total = []
    
    # Training loop
    for epoch in range(num_epochs):
        loss_current_epoch = 0
        for batch_index, (images, labels) in enumerate(train_dataloader):
            # copy images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            
            # forward pass
            y_predicted = model(images)
            loss = loss_fn(y_predicted, labels)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss
            loss_current_epoch += loss.item()

        # At the end of each epoch, save the average loss over batches and display it
        loss_total.append(loss_current_epoch)
        if verbose:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss_current_epoch))
        
    return model, loss_total

#from lab 3
def eval_classifier(model, eval_dataloader, device='cpu', verbose=True):
    model.to(device)
    model.eval() 

    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        # initialize the total and correct number of labels to compute the accuracy
        correct = 0
        total = 0
        for images, labels in eval_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            y_predicted = model(images)
            _, label_predicted = torch.max(y_predicted.data, 1)
            total += labels.size(0)
            correct += (label_predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

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

#Parameters
num_classes = 10
num_epochs = 20
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01

# CNN
num_channels1 = 16
num_channels2 = 32
model_cnn = MusicCNN(num_channels1, num_channels2, num_classes)
model_cnn, loss_total_cnn = training_classifier(model_cnn, train_dataloader, num_epochs, loss_fn, learning_rate, is_mlp=False, device=device, verbose=True)
accuracy_cnn = eval_classifier(model_cnn, test_dataloader, is_mlp=False, device=device, verbose=True)

