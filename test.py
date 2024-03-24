import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from torchvision import datasets, models

if __name__ == '__main__':

    data_loc = r"C:\Users\Arthur King\OneDrive\Documents\My Docs\Class\FINAL SEMESTER\MACHINE LEARNING\Sign Language Project\ASL_Alphabet_Dataset"
    data_loc_train = data_loc + "/asl_alphabet_train/"
    data_loc_test = data_loc + "/asl_alphabet_test"

    # Transformations to apply to each frame
    # Resize frame
    ## No need for rotating images as you won't do sign inverted/rotated
    image_size = (300, 300)  # Arbitrary
    transformations = v2.Compose([
        v2.Resize(image_size),
        v2.RandomHorizontalFlip(p=0.6),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scale to range [0, 1]
        # v2.Normalize(mean=[mean], std=[std])  # Not doing mean and std for dataset
    ])

    BATCH_SIZE = 32
    full_dataset = datasets.ImageFolder(root=data_loc_train, transform=transformations)

    # Calculate the number of images per class you want (half)
    num_samples_per_class = 8000 // 2

    # Generate indices: take the first half from each class
    indices = []
    for class_index in range(len(full_dataset.classes)):
        start_index = class_index * 8000
        end_index = start_index + num_samples_per_class
        indices.extend(range(start_index, end_index))

    # Create a subset based on calculated indices
    train_dataset = Subset(full_dataset, indices)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    t = 0

    for X, y in train_dataloader:
        print(y)
        if t == 10:
            break
        t += 1