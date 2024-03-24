import torch.nn as nn
import model_train
import torch
from torchvision.transforms import v2
import cv2
import time
import pyttsx3


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

def display_thread():
    pass


if __name__ == '__main__':

    labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'delete', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
              13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 'space',
              21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
    len_labels = len(labels)
    print(len_labels)

    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties before adding anything to speak
    engine.setProperty('rate', 100)  # Speed percent (can go over 100)
    engine.setProperty('volume', 1.0)  # Volume 0-1

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

    image_size = (300, 300)  # Arbitrary
    transformations = v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scale to range [0, 1]
        # v2.Normalize(mean=[mean], std=[std])  # Not doing mean and std for dataset
    ])

    inp = int(input("Press 0 key to start capture...."))

    prev_pred_label = None

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

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            #
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            X = preprocess_frame(frame, transformations)

            with torch.no_grad():
                pred = cnn_model(X)
                probabilities = torch.softmax(pred, dim=1)
                max_prob, predicted_val = torch.max(probabilities, dim=1)  # Get the max probability and corresponding label
                # predicted_val = pred.argmax(axis=1).item()

            # Set a confidence threshold
            confidence_threshold = 0.8

            if max_prob > confidence_threshold:
                pred_label = labels[predicted_val.item()]
                if prev_pred_label != pred_label:
                    engine.say(pred_label)
                    # Blocks while processing all the currently queued commands
                    engine.runAndWait()
                print(pred_label)

                prev_pred_label = pred_label

            end_time = time.time()  # Get the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time

            if elapsed_time < frame_delay:  # If the processing was faster than the frame delay, wait
                time.sleep(frame_delay - elapsed_time)

            # # Optionally, display the prediction on the frame
            # label = f"Prediction: {pred_label}"
            # cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #

        # When everything done, release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()