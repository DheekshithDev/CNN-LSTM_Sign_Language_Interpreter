import time

import torch.nn as nn
import model_train
import torch
from torchvision.transforms import v2
import cv2
import torchvision.models.detection as detection


def preprocess_frame(image_frame, transform):
    # Convert the frame color from BGR to RGB
    image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)

    # # Convert the numpy array frame to PIL Image
    # frame_pil = Image.fromarray(frame_rgb)

    # Apply the transformations
    image_frame = transform(image_frame)

    # Add a batch dimension
    image_frame = image_frame.unsqueeze(0)

    # Move to the appropriate device
    image_frame = image_frame.to(DEVICE)

    return image_frame


if __name__ == '__main__':

    labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'delete', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
              13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 'space',
              21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
    len_labels = len(labels)
    print(len_labels)

    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  ## No need for cuda:0 here as it only has one GPU and 0 is default
    print(DEVICE)

    # Initialize the model
    ConvNet = model_train.ConvNet
    cnn_model = ConvNet(input_channels=3, output_classes=28).to(DEVICE)  # Use the same parameters as when you saved the model

    path = r"C:\Users\Arthur King\PycharmProfessionalProjects\SignLanguage"

    # Load the weights
    model_weights_path = path + '/model_weights.pth'
    cnn_model.load_state_dict(torch.load(model_weights_path))
    cnn_model.eval()  ## Set model to eval mode

    # Set
    obj_det_model = detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    obj_det_model.eval()  # Set the model to inference mode

    image_size = (300, 300)  # Arbitrary
    transformations = v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scale to range [0, 1]
        # v2.Normalize(mean=[mean], std=[std])  # Not doing mean and std for dataset
    ])

    inp = int(input("Press 0 key to start capture...."))

    # SETUP DONE # START VIDEO CAPTURE
    if inp == 0:
        # Create a VideoCapture object
        cap = cv2.VideoCapture(0)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        desired_fps = 30  # Set this to whatever FPS you want to simulate
        frame_delay = 1 / desired_fps  # Calculate the delay needed between frames

        while True:
            start_time = time.time()  # Get the start time
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            X = preprocess_frame(frame, transformations)

            with torch.no_grad():
                predictions = obj_det_model(X)

            get_boxes()

            with torch.no_grad():
                pred = cnn_model(X)
                predicted_val = pred.argmax(axis=1).item()

            pred_label = labels[predicted_val]

            print(pred_label)

            end_time = time.time()  # Get the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time

            if elapsed_time < frame_delay:  # If the processing was faster than the frame delay, wait
                time.sleep(frame_delay - elapsed_time)


        # When everything done, release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()