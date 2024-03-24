import sys

import torch.nn as nn
import model_train
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets

if __name__ == '__main__':

    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  ## No need for cuda:0 here as it only has one GPU and 0 is default
    print(DEVICE)

    data_loc = r"C:\Users\Arthur King\OneDrive\Documents\My Docs\Class\FINAL SEMESTER\MACHINE LEARNING\Sign Language Project\ASL_Alphabet_Dataset"
    data_loc_test = data_loc + "/asl_alphabet_test"

    # TRANSFORMATIONS
    image_size = (300, 300)  # Arbitrary
    transformations = v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scale to range [0, 1]
        # v2.Normalize(mean=[mean], std=[std])  # Not doing mean and std for dataset
    ])

    test_dataset = datasets.ImageFolder(root=data_loc_test, transform=transformations)

    test_dataloader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=1)

    # print(len(test_dataloader), len(test_dataloader.dataset))

    # CURR_PATH = model_train.CURR_PATH

    ConvNet = model_train.ConvNet

    # Initialize the model
    cnn_model = ConvNet(input_channels=3, output_classes=28)  # Use the same parameters as when you saved the model

    path = r"C:\Users\Arthur King\PycharmProfessionalProjects\SignLanguage"

    # Load the weights
    model_weights_path = path + '/model_weights.pth'
    cnn_model.load_state_dict(torch.load(model_weights_path))

    # Move the model to the appropriate device
    cnn_model = cnn_model.to(DEVICE)


    # Testing Network
    print("[INFO] Testing network...")

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        cnn_model.eval()  ## Set model to eval mode

        # Store all predictions
        preds = []
        total_predictions = 0

        for X, y in test_dataloader:

            X = X.to(DEVICE)

            # We can make testing faster by testing images in batches just like we did in training
            X = X.unsqueeze(0)  ## If I am testing images one by one without batches

            pred = cnn_model(X)
            predicted_label = pred.argmax(axis=1)
            correct_predictions += (predicted_label == y).sum().item()  # No need for sum() here as it is only one element
            #total_predictions += y.size(0)
            total_predictions += 1
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    print("[INFO] Testing Done.")

    accuracy = correct_predictions / total_predictions
    print(f"[INFO] Testing Done. Accuracy: {accuracy * 100:.2f}%")
    print(preds)
