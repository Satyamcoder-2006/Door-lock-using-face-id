import cv2
import pickle
from utils import trigger_unlock
import time
import datetime
import threading
import numpy as np

class VideoCamera:
    def __init__(self, camera_index=1):
        # Camera setup
        self.camera_index = camera_index
        self.video = None
        self.frame = None
        self.processed_frame = None
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Processing flags and cache
        self.is_running = False
        self.process_this_frame = True
        self.last_recognition_time = 0
        self.recognition_cooldown = 2.0  # Seconds between recognition attempts
        self.last_recognition_result = (False, "Unknown")
        self.last_successful_unlock = 0
        self.unlock_cooldown = 5.0  # Seconds between door unlock attempts
        
        # Set up camera with optimized parameters
        self.connect_camera()
        
        # Load face detector - consider DNN detector for better performance
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load face recognizer model
        self.model = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.model.read('face_recognizer_model.xml')
            print("✅ Face recognition model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading face recognizer model: {e}")
            self.model = None
            
        # Load label map
        try:
            with open('label_map.pkl', 'rb') as f:
                self.label_map = pickle.load(f)
                print("✅ Label map loaded with", len(self.label_map), "users.")
                print("🔑 Authorized users:", ", ".join(self.label_map.values()))
        except Exception as e:
            print(f"❌ Error loading label map: {e}")
            self.label_map = {}
            
        # Start background frame capture thread
        self.start_background_capture()
    
    def connect_camera(self):
        """Try to connect to the camera with optimized settings"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.video = cv2.VideoCapture(self.camera_index)
                if self.video.isOpened():
                    # Set camera properties for better performance
                    self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.video.set(cv2.CAP_PROP_FPS, 30)
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
                    
                    print(f"✅ Camera connected successfully on index {self.camera_index}")
                    return True
                else:
                    raise Exception("Camera failed to open")
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}")
                if self.video:
                    self.video.release()
                time.sleep(1)
                
                # Try alternate camera index if first attempt fails
                if attempt == 0:
                    self.camera_index = 1 if self.camera_index == 0 else 0
        
        print("❌ Failed to connect to any camera")
        return False
    
    def start_background_capture(self):
        """Start a background thread for continuous frame capturing"""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("📷 Background frame capture started")
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        process_every_n_frames = 3  # Process every 3rd frame for recognition
        frame_times = []
        
        while self.is_running:
            if not self.video or not self.video.isOpened():
                if not self.connect_camera():
                    time.sleep(0.5)
                    continue
            
            ret, frame = self.video.read()
            
            if not ret:
                print("⚠️ Failed to grab frame, attempting to reconnect...")
                self.connect_camera()
                continue
            
            # Calculate FPS
            current_time = time.time()
            frame_times.append(current_time)
            
            # Keep only the last 30 frame times for FPS calculation
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            if len(frame_times) > 1:
                self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
            
            # Store raw frame
            self.frame = frame
            
            # Process for face recognition every Nth frame
            if self.frame_count % process_every_n_frames == 0:
                if current_time - self.last_recognition_time > self.recognition_cooldown:
                    self.process_frame(frame.copy())
                    self.last_recognition_time = current_time
            
            self.frame_count += 1
    
    def process_frame(self, frame):
        """Process frame for face recognition"""
        try:
            # Create a smaller frame for faster face detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each face for recognition
            for (x, y, w, h) in faces:
                # Scale coordinates back to original size
                x, y, w, h = x*2, y*2, w*2, h*2
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face for recognition
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                
                # Prepare face for recognition
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (200, 200))
                
                # Apply CLAHE for better lighting normalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                face_gray = clahe.apply(face_gray)
                
                # Recognize face if model is loaded
                if self.model is not None:
                    label_id, confidence = self.model.predict(face_gray)
                    name = self.label_map.get(label_id, "Unknown")
                    
                    # 65 is a good threshold for LBPH (lower is better match)
                    recognized = confidence < 65
                    self.last_recognition_result = (recognized, name)
                    
                    # Show confidence level
                    confidence_text = f"({100 - min(100, confidence):.1f}%)"
                    cv2.putText(frame, f"{name} {confidence_text}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (0, 255, 0) if recognized else (0, 0, 255), 2)
                    
                    # Trigger unlock if recognized and cooldown has passed
                    current_time = time.time()
                    if recognized and current_time - self.last_successful_unlock > self.unlock_cooldown:
                        try:
                            trigger_unlock()
                            self.last_successful_unlock = current_time
                        except Exception as e:
                            print(f"❌ Error triggering unlock: {e}")
                
                # Only process the first face found
                break
            
            # Add door status
            current_time = time.time()
            if current_time - self.last_successful_unlock < 5.0:
                cv2.putText(frame, "Door: Unlocked", (400, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Door: Locked", (400, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp and FPS
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Store processed frame
            self.processed_frame = frame
            
        except Exception as e:
            print(f"❌ Error in frame processing: {e}")
    
    def get_frame(self):
        """Get the latest processed frame as JPEG bytes"""
        if self.processed_frame is None:
            if self.frame is None:
                # Return empty frame if no frames are available
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty_frame, "Connecting to camera...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _, jpeg = cv2.imencode('.jpg', empty_frame)
                return jpeg.tobytes()
            
            # If no processed frame yet, return the raw frame
            frame_copy = self.frame.copy()
            cv2.putText(frame_copy, "Initializing...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame_copy)
            return jpeg.tobytes()
        
        # Return the latest processed frame
        _, jpeg = cv2.imencode('.jpg', self.processed_frame)
        return jpeg.tobytes()
    
    def __del__(self):
        """Clean up resources"""
        self.is_running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.video and self.video.isOpened():
            self.video.release()
            print("📷 Camera released")


def recognize_face(frame, face_cascade, model, label_map):
    """Standalone face recognition function for utils.py replacement"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Apply CLAHE for better lighting normalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_roi = clahe.apply(face_roi)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Predict using the model
            if model is not None:
                label_id, confidence = model.predict(face_roi)
                name = label_map.get(label_id, "Unknown")
                
                # Lower confidence is better in LBPH
                if confidence < 65:  # Threshold for positive recognition
                    return True, name
        
        return False, "Unknown"
        
    except Exception as e:
        print(f"❌ Error in recognize_face: {e}")
        return False, "Error"