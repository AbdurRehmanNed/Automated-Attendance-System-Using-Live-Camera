import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# Define the model path
model_path = 'D:/YOLO/Attendance System/weights/best.pt'  # Replace with your model path

# Load the YOLO model
try:
    model = YOLO(model_path)  # Initialize the YOLO model
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Attendance folder
attendance_folder = 'D:/YOLO/Attendance/'  # Replace with your desired folder path
os.makedirs(attendance_folder, exist_ok=True)

# Student roll numbers and names (example dictionary)
students = {
    "Abdul Samad": "AI-22016",
    "Abdur Rehman": "AI-22014",
    "Sufiyan Nadeem": "AI-22047"
}

# Initialize video capture (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Reduce camera resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize attendance recording
date_str = datetime.now().strftime('%Y-%m-%d')
attendance_file = os.path.join(attendance_folder, f"{date_str}.txt")
with open(attendance_file, 'a+') as file:
    file.seek(0)
    existing_entries = file.read()

# Tracking seen students to avoid duplicates
seen_students = {}
cooldown = 30  # Frames before allowing duplicate entries
frame_counter = 0
frame_skip = 2  # Process every 5th frame

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_counter += 1

        # Skip frames to reduce processing
        if frame_counter % frame_skip != 0:
            continue

        # Run YOLO inference
        results = model(frame, stream=True)  # Use streaming mode for better performance

        # Process detections
        for result in results:
            detections = result.boxes.data  # Bounding box data
            if detections is None:
                continue

            for detection in detections:
                # Extract the class label index and confidence
                class_id = int(detection[5])  # Class index
                confidence = float(detection[4])  # Confidence score

                # Get the label name from YOLO
                label = result.names[class_id]  
                
                # Check if the class corresponds to a known name in the students dictionary
                if label in students and confidence > 0.7:  # Higher confidence threshold
                    roll_no = students[label]
                    entry = f"{roll_no}, {label}\n"

                    # Check cooldown and avoid duplicates
                    if seen_students.get(label, 0) + cooldown < frame_counter:
                        if entry not in existing_entries:  # Avoid duplicate entries in the file
                            with open(attendance_file, 'a') as file:
                                file.write(entry)
                            existing_entries += entry
                            print(f"Added attendance entry: {entry.strip()}")
                        seen_students[label] = frame_counter

        # Display the resulting frame with bounding boxes
        annotated_frame = result.plot()  # Annotate the frame with detections
        cv2.imshow('Live Camera - Attendance System', annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error processing the live feed: {e}")

finally:
    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
