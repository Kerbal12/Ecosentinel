import cv2
import numpy as np

# Load YOLO model files
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get the names of all YOLO layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (from coco.names)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# List of vehicle classes and the person class for detection
vehicle_classes = ["car", "motorbike", "bus", "truck"]
person_class = "person"

# Ask user to choose input method
choice = input("Enter 1 to use video file or 2 to use webcam: ")

if choice == '1':
    video_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_path)
elif choice == '2':
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
else:
    print("Invalid choice. Exiting...")
    exit()

# Flag to ensure only one alert frame is saved
alert_triggered = False

while True:
    ret, frame = cap.read()  # Capture each frame
    if not ret:
        break

    # Get dimensions of the frame
    h, w = frame.shape[:2]

    # Prepare the image as input for the neural network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform the forward pass to get the detections
    detections = net.forward(output_layers)

    # Initialize lists to hold detection results
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections by confidence threshold
            if confidence > 0.5:
                center_x = int(obj[0] * w)
                center_y = int(obj[1] * h)
                width = int(obj[2] * w)
                height = int(obj[3] * h)

                # Get top-left corner of the bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if indices are not empty and loop over the detected objects
    if len(indices) > 0:
        for i in indices.flatten():  # Use flatten correctly
            x, y, width, height = boxes[i]
            class_name = classes[class_ids[i]]

            # Check if class_name is a vehicle or person
            if class_name in vehicle_classes or class_name == person_class:
                color = (0, 0, 255) if class_name in vehicle_classes else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                label = f"{class_name}: {confidences[i]:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save an alert frame and display a message
                if not alert_triggered:
                    alert_triggered = True
                    cv2.imwrite("alert_frame.jpg", frame)
                    print("Alert: Possible poaching detected! Frame saved as 'alert_frame.jpg'.")

    # Show the processed frame
    cv2.imshow('Vehicle and Human Detection (YOLO)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
