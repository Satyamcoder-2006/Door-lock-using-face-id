import cv2
import numpy as np
import pickle
import time
from utils import connect_arduino, enhance_image

# === Arduino Setup ===
arduino = connect_arduino()

# === Load Face Recognition Model and Label Map ===
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognizer_model.xml')

with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# === Start Webcam ===
cap = cv2.VideoCapture(1)  # Change to 0 if using built-in camera

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

print("🚀 Starting Face Recognition Door Lock System...")
print("📸 Camera initialized")
print("🧠 Face recognition model loaded")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        time.sleep(0.5)  # Wait a bit before retrying
        continue

    # Apply image enhancement
    enhanced = enhance_image(frame)
    
    # MODIFIED: Improved face detection parameters
    faces = face_classifier.detectMultiScale(
        enhanced,
        scaleFactor=1.05,  # Reduced from 1.3
        minNeighbors=3,    # Reduced from 5
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        cv2.putText(frame, "🔍 No Face Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Sort faces by size (use the largest)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        face_gray = enhanced[y:y + h, x:x + w]
        face_resized = cv2.resize(face_gray, (200, 200))

        # MODIFIED: Apply additional preprocessing
        face_resized = cv2.equalizeHist(face_resized)  # Improve contrast

        label, confidence = model.predict(face_resized)
        name = label_map.get(label, "Unknown")

        # MODIFIED: Added debug information
        print(f"🧠 Predicted: {name}, Label: {label}, Confidence: {round(confidence, 2)}")

        # MODIFIED: Increased confidence threshold
        if confidence < 105:  # Increased from 85
            cv2.putText(frame, f"✅ {name} ({round(confidence, 2)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            print(f"🟢 Face recognized as {name}. Sending unlock signal to Arduino.")
            if arduino:
                try:
                    # MODIFIED: Send unlock signal multiple times to ensure receipt
                    for _ in range(3):  
                        arduino.write(b'UNLOCK\n')
                        arduino.flush()
                        time.sleep(0.1)
                except Exception as e:
                    print(f"⚠️ Error sending data to Arduino: {e}")

            cv2.putText(frame, "🔓 ACCESS GRANTED", (30, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Verified", frame)
            cv2.waitKey(5000)  # Wait before closing
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Door unlocked successfully")
            exit()

        else:
            cv2.putText(frame, f"❌ Unknown ({round(confidence, 2)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # MODIFIED: Added more feedback about the confidence level
            if confidence < 130:
                cv2.putText(frame, "Almost recognized - try better lighting", (x, y + h + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)

    cv2.imshow("Face Verifier", frame)

    key = cv2.waitKey(1)
    if key == 13:  # Press Enter to exit
        break
    elif key == 114 or key == 82:  # Press 'r' or 'R' to retry with current frame
        print("🔄 Retrying recognition with current frame...")
        # The loop will naturally retry on the next iteration

cap.release()
cv2.destroyAllWindows()