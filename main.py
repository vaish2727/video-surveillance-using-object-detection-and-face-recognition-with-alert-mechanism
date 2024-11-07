import cv2
import torch
import face_recognition
import numpy as np
import os
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "recepient@gmail.com"
SMTP_PASSWORD = "password"
RECIPIENT_EMAIL = "receiver@email.com"

def send_alert(frame, detected_objects, unauthorized_faces):
    # Save the frame as an image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_image_path = f"alert_{timestamp}.jpg"
    cv2.imwrite(alert_image_path, frame)
    
    # Create email
    msg = MIMEMultipart()
    msg['Subject'] = 'Security Alert - Suspicious Activity Detected'
    msg['From'] = SMTP_USERNAME
    msg['To'] = RECIPIENT_EMAIL
    
    text = f"""Security Alert!
    
Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Detected Objects: {', '.join(detected_objects)}
Unauthorized Faces: {len(unauthorized_faces)}
    """
    msg.attach(MIMEText(text))
    
    # Attach image
    with open(alert_image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)
    
    # Send email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print("Alert sent successfully")
    except Exception as e:
        print(f"Failed to send alert: {e}")
    
    # Clean up
    os.remove(alert_image_path)

# Alert configuration
ALERT_OBJECTS = ["person", "car", "truck"]
ALERT_COOLDOWN = 60
last_alert_time = datetime.datetime.now() - datetime.timedelta(seconds=ALERT_COOLDOWN)

# Load YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes = model.names 
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names

try:
    known_face_encodings, known_face_names = load_known_faces("known_faces")
except Exception as e:
    print(f"Error loading known faces: {e}")
    known_face_encodings, known_face_names = [], []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ensure frame is in RGB format for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    height, width, _ = frame.shape
    
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []
    unauthorized_faces = []
    
    # Object detection
    results = model(frame)
    detections = results.xyxy[0].numpy()  # get detections
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.5:  # confidence threshold
            class_ids.append(int(cls))
            confidences.append(float(conf))
            detected_objects.append(classes[int(cls)])
        # Convert to x, y, w, h format for drawing
            width, height = x2 - x1, y2 - y1
            boxes.append([int(x1), int(y1), int(width), int(height)])
    # Face detection using the RGB frame
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding) if known_face_encodings else []
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                unauthorized_faces.append((top, right, bottom, left))
            
            # Draw rectangle and name (convert back to BGR for OpenCV)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error in face detection: {e}")
    
    # Draw object detection boxes
    for i, box in enumerate(boxes):
        if class_ids[i] in range(len(classes)):  # check if class_id is valid
            x, y, w, h = box
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Check if alert should be triggered
    current_time = datetime.datetime.now()
    if (detected_objects or unauthorized_faces) and \
       (current_time - last_alert_time).seconds >= ALERT_COOLDOWN:
        send_alert(frame, detected_objects, unauthorized_faces)
        last_alert_time = current_time
    
    cv2.imshow("Video Surveillance", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()