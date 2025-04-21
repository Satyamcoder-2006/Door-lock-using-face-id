import cv2
import numpy as np
import pickle
import time
import os
from utils import connect_arduino, enhance_image

# === Arduino Setup ===
arduino = connect_arduino()

# === Load DNN Face Detector ===
prototxt_path = os.path.join('models', 'deploy.prototxt')
model_path = os.path.join('models', 'res10_300x300_ssd_iter_140000.caffemodel')

# If model files exist, use DNN face detector, otherwise fallback to Haar
if os.path.exists(prototxt_path) and os.path.exists(model_path):
    print("📁 Loading DNN face detector...")
    face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    use_dnn = True
else:
    print("📁 DNN models not found, using Haar Cascade...")
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    use_dnn = False

# === Load Face Recognition Model and Label Map ===
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognizer_model.xml')

with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# === Initialize face landmark detector if available ===
try:
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("models/lbfmodel.yaml")
    use_landmarks = True
    print("📁 Face landmark detector loaded successfully")
except:
    use_landmarks = False
    print("⚠️ Face landmark detector not available")

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

# === Track face recognition history ===
recent_recognitions = []
recognition_history_size = 5

# === Functions for face alignment ===
def align_face(image, landmarks):
    try:
        # Get the landmark points for the detected face
        points = landmarks[0][0]
        
        # Calculate eye centers
        left_eye = points[36:42].mean(axis=0).astype("int")
        right_eye = points[42:48].mean(axis=0).astype("int")
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get image dimensions
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned
    except:
        return image

# === Function to detect faces using DNN ===
def detect_faces_dnn(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Minimum confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x2, y2) = box.astype("int")
            w = x2 - x
            h = y2 - y
            faces.append((x, y, w, h))
    
    return faces

# === Main recognition loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        time.sleep(0.5)  # Wait a bit before retrying
        continue

    # Apply image enhancement
    enhanced = enhance_image(frame)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) > 2 else enhanced
    
    # Detect faces
    if use_dnn:
        faces = detect_faces_dnn(enhanced)
    else:
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
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
        
        # Extract face region
        face_roi = gray[y:y + h, x:x + w]
        
        # Apply face alignment if landmarks are available
        if use_landmarks:
            success, landmarks = landmark_detector.fit(gray, np.array([faces[0]]))
            if success:
                face_roi = align_face(face_roi, landmarks)
        
        # Prepare face for recognition
        face_resized = cv2.resize(face_roi, (200, 200))
        face_processed = cv2.equalizeHist(face_resized)
        
        # Check lighting conditions and adjust preprocessing
        brightness = cv2.mean(face_processed)[0]
        if brightness < 80:  # Low light condition
            face_processed = cv2.convertScaleAbs(face_processed, alpha=1.5, beta=30)
        elif brightness > 200:  # Bright light condition
            face_processed = cv2.convertScaleAbs(face_processed, alpha=0.8, beta=-10)
            
        # Apply additional preprocessing
        face_processed = cv2.GaussianBlur(face_processed, (5, 5), 0)

        # Recognize face
        label, confidence = model.predict(face_processed)
        name = label_map.get(label, "Unknown")

        # Add to recognition history
        recent_recognitions.append((name, confidence))
        if len(recent_recognitions) > recognition_history_size:
            recent_recognitions.pop(0)
        
        # Get most frequent recognition
        if len(recent_recognitions) >= 3:
            names = [r[0] for r in recent_recognitions]
            most_common_name = max(set(names), key=names.count)
            avg_confidence = sum([r[1] for r in recent_recognitions if r[0] == most_common_name]) / names.count(most_common_name)
        else:
            most_common_name = name
            avg_confidence = confidence

        # Debug information
        print(f"🧠 Predicted: {name}, Label: {label}, Confidence: {round(confidence, 2)}")
        print(f"📊 Recent recognitions: {recent_recognitions}")
        print(f"🔍 Most likely: {most_common_name} (Avg conf: {round(avg_confidence, 2)})")

        # Adjust confidence threshold based on lighting
        confidence_threshold = 95 if 80 <= brightness <= 200 else 105
        
        # Check if face is recognized
        if avg_confidence < confidence_threshold and most_common_name != "Unknown":
            cv2.putText(frame, f"✅ {most_common_name} ({round(avg_confidence, 2)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            print(f"🟢 Face recognized as {most_common_name}. Sending unlock signal to Arduino.")
            if arduino:
                try:
                    # Send unlock signal multiple times to ensure receipt
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
            status_text = f"❌ Unknown ({round(confidence, 2)})" if most_common_name == "Unknown" else f"❌ Not Confident: {most_common_name} ({round(avg_confidence, 2)})"
            cv2.putText(frame, status_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Provide feedback about the confidence level
            if avg_confidence < 130:
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