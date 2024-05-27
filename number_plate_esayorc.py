# Copyright 2024 Rashee Wijesinghe

import cv2
import easyocr
from ultralytics import YOLO

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the model
model = YOLO('NumberPlate.pt') 

# Function to recognize text from image using EasyOCR
def recognize_text(plate_img):
    # Use EasyOCR to recognize text
    results = reader.readtext(plate_img)
    text = ''
    for result in results:
        text += result[1] + ' '
    return text.strip()


# Open the video file
cap = cv2.VideoCapture('\path\to\video')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the YOLO model to detect number plates
    results = model(frame, conf=0.3)
    
    for result in results:
        # Get bounding box coordinates
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop the detected license plate
            plate_img = frame[y1:y2, x1:x2]
            
            # Recognize text from the cropped image
            plate_number = recognize_text(plate_img)
            
            # Calculate text size
            text_size, _ = cv2.getTextSize(plate_number, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
            text_w, text_h = text_size
            
            # Coordinates for the text background rectangle
            rect_x1, rect_y1 = x1-10, y1 - 20 - text_h
            rect_x2, rect_y2 = x1 +10 + text_w, y1 
            
            # Draw the background rectangle
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)
            
            # Draw the text over the rectangle
            cv2.putText(frame, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

    # Display the frame with bounding boxes and recognized text
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
