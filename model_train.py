# IMPORTS
import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from torchvision import datasets, models
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
# from sklearn.metrics import classification_report
import time
from torch.cuda.amp import autocast, GradScaler

####
import cv2
import random
from PIL import Image


class ConvNet(nn.Module):
    def __init__(self, input_channels, output_classes):
        super().__init__()

        self.resnet_cnn_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # Replace the final fully connected layer (fc) to match the number of output_classes
        num_features = self.resnet_cnn_model.fc.in_features  # Get the input feature size of the original fc layer
        self.resnet_cnn_model.fc = nn.Linear(num_features, output_classes)  # Replace the fc layer

        # self.cnn_model = nn.Sequential(
        #     nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=(6, 6)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),  ## the dimensions of the input feature map should be multiples of the stride for better performance so the stride wont miss some corner parts of the feature map grid
        #     nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(5, 5)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        #     nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(4, 4)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  ## stride can be none as it defaults to kernel_size when none
        #     nn.Conv2d(in_channels=384, out_channels=768, kernel_size=(3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  ## stride can be none as it defaults to kernel_size when none
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
        #
        # # This snippet is for programmatically determining the cnn output to serve as input for LSTM
        # dummy_input = torch.randn(1, input_channels, 300, 300)  ## [batch_size, channels, height, width]
        # dummy_output = self.cnn_model(dummy_input)
        # cnn_output = dummy_output.view(dummy_output.size(0), -1).size(1)
        # #cnn_output = dummy_output.size(1)
        #
        # self.fc_model = nn.Sequential(
        #     nn.Linear(in_features=cnn_output, out_features=62),
        #     nn.ReLU(),
        #     nn.Linear(in_features=62, out_features=48),
        #     nn.ReLU(),
        #     nn.Linear(in_features=48, out_features=output_classes)
        #     ## Need to change the sizes of hyperparameters as the output_classes number exceed the in_features at this current level
        #     # nn.LogSoftmax(dim=1)  ## No need for any final activation here if I'm using CrossEntropyLoss which internally handles 'Softmax' + 'NLLLoss' in single operation
        # )

    def forward(self, x):
        batch_size, c, h, w = x.size()

        x = self.resnet_cnn_model(x)

        # x = self.cnn_model(x)
        # # x.view(x.size(0), -1). This is alternative to flatten()
        # x = torch.flatten(x, start_dim=1)  ## Maybe use nn.flatten() (what's the diff.)
        # x = self.fc_model(x)
        return x


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))

    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  ## No need for cuda:0 here as it only has one GPU and 0 is default
    print(DEVICE)

    # Set RNGs to same values every time including CUDA operations
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) ## For multiple GPUs

    data_loc = r"C:\Users\Arthur King\OneDrive\Documents\My Docs\Class\FINAL SEMESTER\MACHINE LEARNING\Sign Language Project\ASL_Alphabet_Dataset"
    data_loc_train = data_loc + "/asl_alphabet_train/"
    data_loc_test = data_loc + "/asl_alphabet_test"

    # Transformations to apply to each frame
    # Resize frame
    ## No need for rotating images as you won't do sign inverted/rotated
    image_size = (500, 500)  # Arbitrary
    transformations = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.RandomRotation(degrees=(0, 360)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scale to range [0, 1]
        # v2.Normalize(mean=[mean], std=[std])  # Not doing mean and std for dataset
    ])

    BATCH_SIZE = 32
    full_dataset = datasets.ImageFolder(root=data_loc_train, transform=transformations)

    # # Calculate the number of images per class you want (half)
    # num_samples_per_class = 8000 // 2
    #
    # # Generate indices: take the first half from each class
    # indices = []
    # for class_index in range(len(full_dataset.classes)):
    #     start_index = class_index * 8000
    #     end_index = start_index + num_samples_per_class
    #     indices.extend(range(start_index, end_index))

    #
    # # Create a subset based on calculated indices
    # train_dataset = Subset(full_dataset, indices)

    train_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # BATCH_SIZE = 32
    # train_dataset = datasets.ImageFolder(root=data_loc_train, transform=transformations) # This itself is converting X to float32, y to int64
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


    ######  MODEL CREATION  ######
    input_channels = 3  # RGB
    output_classes = 28  # ASL # 28 is num of alphabets available in the dataset
    cnn_model = ConvNet(input_channels, output_classes).to(DEVICE)
    print(cnn_model)
    # print(next(cnn_model.parameters()).device)

    # Loss function and Optimizer
    #scaler = GradScaler()
    LRN_RATE = 0.001
    loss_function = nn.CrossEntropyLoss()
    g_descent_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LRN_RATE)  ##Adam is a type of gradient descent


    ######  TRAINING MODEL  ######
    overall_train_losses = []
    overall_train_accuracy = []
    # test_losses = []
    # test_correct = []

    EPOCHS = 10
    # total_dataset = len(train_dataloader.dataset)

    print("[INFO] Training the network...")
    start_time = time.time()

    for e in range(EPOCHS):
        cnn_model.train()  # Set model to train mode

        # These variables are for entire dataset once (all batches)
        total_train_loss = 0
        total_train_correct = 0

        for X_batch, y_batch in train_dataloader:

            # Push Data Tensors to the GPU
            (X_batch, y_batch) = X_batch.to(DEVICE), y_batch.to(DEVICE)

            g_descent_optimizer.zero_grad()

            # with autocast():
            #     pred = cnn_model(X_batch)  # Predicted values of y
            #     loss = loss_function(pred, y_batch)
            #
            # scaler.scale(loss).backward()
            # scaler.step(g_descent_optimizer)
            # scaler.update()

            pred = cnn_model(X_batch)  # Predicted values of y
            loss = loss_function(pred, y_batch)

            #g_descent_optimizer.zero_grad()
            loss.backward()
            g_descent_optimizer.step()

            total_train_loss += loss
            total_train_correct += (pred.argmax(1) == y_batch).type(torch.float).sum().item()

        # For one epoch
        avg_train_accuracy = total_train_correct / len(train_dataloader.dataset)
        avg_train_loss = total_train_loss / len(train_dataloader)

        overall_train_accuracy.append(avg_train_accuracy)
        overall_train_losses.append(avg_train_loss)

        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, avg_train_accuracy))
        print(f'{total_train_correct}/{len(train_dataloader.dataset)}')

    end_time = time.time()

    total_time = end_time - start_time

    print(f'Total Training Time: {total_time / 60} Minutes.')

    CURR_PATH = os.getcwd()
    print(f'Current working directory: {CURR_PATH}')

    torch.save(cnn_model.state_dict(), 'model_weights.pth')





