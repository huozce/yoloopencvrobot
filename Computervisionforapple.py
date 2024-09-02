import os
from ultralytics import YOLO
import cv2
import serial

# Arduino serial communication setup
ser = serial.Serial('COM3', 115200)  # Replace 'COM3' with your Arduino's serial port

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust if you have multiple cameras

model_path = "C:\\Users\\husoc\\runs\\detect\\train29\\weights\\best.pt" #Use your custom model.

# Load your YOLO model for apple detection
model = YOLO(model_path)

# Adjust the hue range for yellowish, reddish, and greenish colors
hue_lower_apple = 10  # Lower bound for apple hue
hue_upper_apple = 40  # Upper bound for apple hue

hue_lower_shirt = 100  # Lower bound for shirt hue
hue_upper_shirt = 130  # Upper bound for shirt hue



hue_lower_green = 40   # Lower bound for green hue
hue_upper_green = 80   # Upper bound for green hue

hue_lower_yellow = 20  # Lower bound for yellow hue
hue_upper_yellow = 30  # Upper bound for yellow hue

threshold = 0.01  # Confidence threshold for considering an object as an apple

# Define size thresholds
min_size_threshold = 5  # Adjust the minimum size as needed
max_size_threshold = 20000  # Adjust the maximum size as needed

max_confidence = 0  # Maximum confidence among detected objects
coordinates_to_send = None  # Coordinates to be sent to Arduino

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define masks for apple, shirt, green, and yellow colors
    mask_apple = cv2.inRange(hsv_frame, (hue_lower_apple, 50, 50), (hue_upper_apple, 255, 255))
    mask_shirt = cv2.inRange(hsv_frame, (hue_lower_shirt, 50, 50), (hue_upper_shirt, 255, 255))
    mask_green = cv2.inRange(hsv_frame, (hue_lower_green, 50, 50), (hue_upper_green, 255, 255))
    mask_yellow = cv2.inRange(hsv_frame, (hue_lower_yellow, 50, 50), (hue_upper_yellow, 255, 255))

    # Combined mask for apple, shirt, green, and yellow colors
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(mask_apple, mask_shirt), mask_green), mask_yellow)

    # Apply the combined mask to the frame
    filtered_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Perform inference using the loaded model on the filtered frame
    results = model(filtered_frame)[0]

    # Reset values for each iteration
    max_confidence = 0
    coordinates_to_send = None

    # Detect only one object at a time (highest confidence)
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        box_size = (x2 - x1) * (y2 - y1)

        if score > threshold and min_size_threshold < box_size < max_size_threshold:
            # Calculate the center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Boost confidence for green color
            if hue_lower_green <= (hue_lower_apple + hue_upper_apple) / 2 <= hue_upper_green:
                score += 0.5

            # Boost confidence for yellow color
            if hue_lower_yellow <= (hue_lower_apple + hue_upper_apple) / 2 <= hue_upper_yellow:
                score += 0.5

            # Update the maximum confidence and coordinates
            if score > max_confidence:
                max_confidence = score
                coordinates_to_send = (center_x, center_y)

    # If there are coordinates to send, send them to the Arduino
    if coordinates_to_send is not None:
        data_to_send = f"{int(coordinates_to_send[0])},{int(coordinates_to_send[1])}\n"
        ser.write(data_to_send.encode())

    # Draw square around the detected object and display coordinates and confidence
    if coordinates_to_send is not None:
        x, y = coordinates_to_send
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        label = f"Coordinates: ({int(x)}, {int(y)}) | Confidence: {max_confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()