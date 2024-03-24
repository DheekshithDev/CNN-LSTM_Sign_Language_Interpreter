import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if video opened successfully
if not cap.isOpened():
  print("Error: Could not open video.")
  exit()

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # If frame is read correctly ret is True
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

  # Optionally, display the prediction on the frame
  #label = f"Prediction: {pred_label}"
  #cv2.putText(frame, 1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

  # Display the resulting frame
  cv2.imshow('Frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()