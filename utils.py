import cv2
import serial
import time
import numpy as np
from serial.tools import list_ports
import atexit

# Global Arduino reference
arduino = None
arduino_connected = False

def cleanup_arduino():
    global arduino, arduino_connected
    if arduino is not None:
        try:
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            arduino.flush()
            arduino.close()
            print("🔌 Arduino connection closed properly")
        except Exception as e:
            print(f"⚠️ Error during Arduino cleanup: {e}")
        finally:
            arduino = None
            arduino_connected = False
            time.sleep(2)

atexit.register(cleanup_arduino)

def find_arduino_port():
    arduino_ports = []
    for port in list_ports.comports():
        if "Arduino" in port.description or "CH340" in port.description or "USB Serial" in port.description:
            arduino_ports.append(port.device)
        elif "COM" in port.device:
            arduino_ports.append(port.device)

    if not arduino_ports:
        print("⚠️ No potential Arduino ports found")
        return None

    print(f"📝 Found potential Arduino ports: {', '.join(arduino_ports)}")
    return arduino_ports

def try_all_available_ports(baudrate=9600):
    ports = find_arduino_port()
    if not ports:
        return None

    for port in ports:
        arduino = connect_arduino(port, baudrate)
        if arduino:
            return arduino
    return None

def connect_arduino(port='COM7', baudrate=9600):
    global arduino, arduino_connected
    cleanup_arduino()
    print(f"🔌 Attempting to connect to {port}...")

    for attempt in range(5):
        try:
            arduino = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1,
                write_timeout=1,
                exclusive=True
            )
            time.sleep(2)
            arduino.write(b't')
            time.sleep(0.5)
            response = arduino.read(size=1)

            if response:
                print(f"✅ Arduino connected successfully on {port}")
                arduino_connected = True
                return arduino
            else:
                print(f"⚠️ No response from Arduino on attempt {attempt + 1}")
                cleanup_arduino()

        except serial.SerialException as e:
            print(f"⚠️ Connection attempt {attempt + 1} failed: {str(e)}")
            cleanup_arduino()
            continue

    print("❌ Could not establish connection to Arduino")
    arduino_connected = False
    return None

# ✅ FIXED: ArduinoManager now exposes write() and flush()
class ArduinoManager:
    def __init__(self, port='COM7', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = connect_arduino(self.port, self.baudrate)  # 💥 INIT RIGHT HERE

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_arduino()

    def write(self, data):
        if self.ser:
            try:
                self.ser.write(data)
                self.ser.flush()
                print(f"✅ Successfully wrote {data} to Arduino")
                return True
            except Exception as e:
                print(f"❌ Error writing to Arduino: {e}")
                return False
        else:
            print("❌ Cannot write to Arduino: No connection")
            return False

    def read(self, size=1):
        if self.ser:
            try:
                response = self.ser.read(size)
                if response:
                    print(f"✅ Read from Arduino: {response}")
                return response
            except Exception as e:
                print(f"❌ Error reading from Arduino: {e}")
                return None
        return None

    def flush(self):
        if self.ser:
            try:
                self.ser.flush()
                return True
            except Exception as e:
                print(f"❌ Error flushing Arduino buffer: {e}")
                return False
        return False

# Pre-create CLAHE object and kernel for better performance
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

def enhance_image(frame, fast_mode=True):
    """Enhance image for face detection with optional fast mode"""
    if len(frame.shape) > 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Fast mode for real-time processing
    if fast_mode:
        # Apply essential enhancements for better face detection
        enhanced = _clahe.apply(gray)
        # Apply a light Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        # Apply basic sharpening to enhance edges
        enhanced = cv2.filter2D(enhanced, -1, _sharpening_kernel)
        return enhanced

    # Full enhancement for training and user addition
    enhanced = _clahe.apply(gray)
    # Apply bilateral filter to smooth while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    # Apply sharpening to enhance facial features
    enhanced = cv2.filter2D(enhanced, -1, _sharpening_kernel)
    # Normalize the image to improve contrast
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    return enhanced

def recognize_face(frame, face_classifier=None, model=None, label_map=None):
    """Optimized face recognition function"""
    if not all([face_classifier, model, label_map]):
        print("⚠️ Missing required components for face recognition")
        return False, "Error"

    try:
        # Use fast mode for real-time recognition
        enhanced = enhance_image(frame, fast_mode=True)

        # Optimized face detection parameters
        faces = face_classifier.detectMultiScale(
            enhanced,
            scaleFactor=1.1,      # Faster detection
            minNeighbors=4,      # More reliable detection
            minSize=(50, 50),    # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return False, "No Face"

        # Get the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        # Process face for recognition
        face = enhanced[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)  # Histogram equalization improves recognition

        # Perform recognition
        label, confidence = model.predict(face)
        name = label_map.get(label, "Unknown")

        # Adjusted threshold for better accuracy
        threshold = 100
        if confidence < threshold:
            print(f"🟢 Face recognized: {name} | Confidence: {confidence}")
            return True, name
        else:
            print(f"❌ Face not recognized | Confidence: {confidence}")
            return False, "Unknown"

    except Exception as e:
        print(f"⚠️ Error during face recognition: {str(e)}")
        return False, "Error"

# ✅ Now works with the new ArduinoManager interface
def trigger_unlock():
    with ArduinoManager() as arduino:
        if not arduino:
            print("⚠️ Arduino connection failed")
            return False
        try:
            arduino.write(b'u')
            time.sleep(0.2)
            print("🔓 Unlock signal sent")
            return True
        except Exception as e:
            print(f"❌ Failed to send unlock signal: {e}")
            return False
