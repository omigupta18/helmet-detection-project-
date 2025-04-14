import cv2
import torch
import matplotlib.pyplot as plt

# Load pretrained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Render predictions on the frame
    annotated_frame = results.render()[0]

    # Display
    plt.imshow(annotated_frame[..., ::-1])
    plt.title("YOLOv5 - webcam detection")
    plt.pause(0.001)
    cv2.imshow("webcam",frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
