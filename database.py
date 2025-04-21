import cv2
import os
import shutil
import numpy as np
import time
from datetime import datetime
import math

# Load the Haar cascade
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up DNN face detector if available
try:
    models_dir = "models"
    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if os.path.exists(prototxt_path) and os.path.exists(model_path):
        print("📁 Loading DNN face detector...")
        face_detector_dnn = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        use_dnn = True
    else:
        use_dnn = False
except:
    use_dnn = False
    print("⚠️ DNN face detector not available, using Haar Cascade")

# Face extraction function with improved detection
def face_extractor(img, quality_mode=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if use_dnn:
        # DNN-based detection
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector_dnn.setInput(blob)
        detections = face_detector_dnn.forward()
        
        # Find face with highest confidence
        best_confidence = 0
        best_face = None
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Minimum confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Calculate width and height
                face_w = x2 - x
                face_h = y2 - y
                
                # Only process if face dimensions are reasonable
                if face_w > 30 and face_h > 30:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        face_roi = img[y:y2, x:x2]
                        best_face = face_roi
        
        if best_face is not None:
            # For quality mode, return confidence score too
            if quality_mode:
                return best_face, best_confidence
            return best_face
    
    # Fallback to Haar cascade
    faces = face_classifier.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        if quality_mode:
            return None, 0
        return None
    
    # Sort faces by size and use the largest
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]
    face_roi = img[y:y+h, x:x+w]
    
    if quality_mode:
        # For haar cascade, we'll estimate quality based on face size
        quality_score = min(0.9, (w * h) / (img.shape[0] * img.shape[1] * 4))
        return face_roi, quality_score
    
    return face_roi

# Image augmentation functions
def apply_augmentations(image):
    augmented_images = []
    
    # Original image
    augmented_images.append(("original", image))
    
    # Brightness variations
    for factor in [0.8, 1.2]:
        brightened = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append((f"brightness_{factor}", brightened))
    
    # Rotation variations
    for angle in [-10, 10]:
        rotated = rotate_image(image, angle)
        augmented_images.append((f"rotate_{angle}", rotated))
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(("flip", flipped))
    
    # Slight zoom
    h, w = image.shape[:2]
    zoom_factor = 0.9
    crop_h, crop_w = int(h * zoom_factor), int(w * zoom_factor)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    zoomed = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    zoomed = cv2.resize(zoomed, (w, h))
    augmented_images.append(("zoom", zoomed))
    
    return augmented_images

def rotate_image(image, angle):
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

# Base folder for images
base_path = r"C:\Users\satya\OneDrive\Desktop\IOTPROJECTFINAL\userimage"
augmented_path = r"C:\Users\satya\OneDrive\Desktop\IOTPROJECTFINAL\userimage_augmented"

# Create augmented directory if it doesn't exist
os.makedirs(augmented_path, exist_ok=True)

def add_user():
    user_name = input("Enter new user name: ").strip()
    save_path = os.path.join(base_path, f"User_{user_name}")
    os.makedirs(save_path, exist_ok=True)
    
    # Create augmented directory for this user
    aug_save_path = os.path.join(augmented_path, f"User_{user_name}")
    os.makedirs(aug_save_path, exist_ok=True)

    cap = cv2.VideoCapture(1)
    count = 0
    max_images = 100
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    # Variables for guidance
    start_time = time.time()
    instructions = [
        "Look straight at the camera",
        "Turn slightly to the left",
        "Turn slightly to the right",
        "Tilt your head up slightly",
        "Tilt your head down slightly",
        "Move closer to the camera",
        "Move farther from the camera",
        "Change lighting (if possible)"
    ]
    current_instruction = 0
    instruction_change_interval = 8  # seconds
    last_quality_check = time.time()
    quality_check_interval = 0.5  # seconds
    
    face_quality_threshold = 0.6
    high_quality_count = 0
    consecutive_no_face = 0
    max_consecutive_no_face = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
            
        original_frame = frame.copy()
        
        # Check if it's time to update instructions
        if time.time() - start_time > instruction_change_interval:
            current_instruction = (current_instruction + 1) % len(instructions)
            start_time = time.time()
        
        current_guide = instructions[current_instruction]
        
        # Face quality detection (less frequent to save processing)
        face_detected = False
        if time.time() - last_quality_check > quality_check_interval:
            face_result, quality = face_extractor(frame, quality_mode=True)
            last_quality_check = time.time()
            
            if face_result is not None:
                face_detected = True
                consecutive_no_face = 0
                
                # Save high-quality faces
                if quality > face_quality_threshold:
                    count += 1
                    high_quality_count += 1
                    face = cv2.resize(face_result, (200, 200))
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    
                    # Save original face
                    timestamp = datetime.now().strftime("%H%M%S%f")
                    file_name = os.path.join(save_path, f"{user_name}_{count}_{timestamp}.jpg")
                    cv2.imwrite(file_name, face_gray)
                    
                    # Create and save augmentations
                    augmentations = apply_augmentations(face)
                    for aug_name, aug_img in augmentations:
                        if aug_name != "original":  # Skip original as we already saved it
                            aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                            aug_file = os.path.join(aug_save_path, f"{user_name}_{count}_{aug_name}_{timestamp}.jpg")
                            cv2.imwrite(aug_file, aug_gray)
                    
                    print(f"✅ Saved face {count}/{max_images} (Quality: {quality:.2f})")
            else:
                consecutive_no_face += 1
        
        # Display to user
        display_frame = original_frame.copy()
        
        # Show guidance and progress
        progress_text = f"Progress: {count}/{max_images} faces"
        cv2.putText(display_frame, progress_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show current instruction
        instruction_text = f"Please: {current_guide}"
        cv2.putText(display_frame, instruction_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show face detection status
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        status_text = "Face Detected" if face_detected else "No Face Detected"
        cv2.putText(display_frame, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show augmentation info
        if count > 0:
            aug_text = f"Created {(count-1)*5 + high_quality_count} augmented samples"
            cv2.putText(display_frame, aug_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        cv2.imshow("Face Data Collection", display_frame)
        
        # Exit conditions
        key = cv2.waitKey(1)
        if key == 13 or count >= max_images:  # Enter key
            break
            
        # Too many consecutive frames without a face
        if consecutive_no_face > max_consecutive_no_face:
            print("⚠️ No face detected for too long. Please position yourself in front of the camera.")
            consecutive_no_face = 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collection complete! Collected {count} original face samples for {user_name}")
    print(f"✨ Created approximately {count*5} augmented samples for better training")

def delete_user():
    users = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not users:
        print("⚠️ No users found.")
        return

    print("\nAvailable Users:")
    for idx, user in enumerate(users):
        print(f"{idx + 1}. {user}")

    try:
        choice = int(input("Enter the number of the user to delete: "))
        if 1 <= choice <= len(users):
            user_to_delete = users[choice - 1]
            confirm = input(f"Are you sure you want to delete '{user_to_delete}'? (y/n): ").strip().lower()
            if confirm == 'y':
                # Delete from original dataset
                orig_path = os.path.join(base_path, user_to_delete)
                if os.path.exists(orig_path):
                    shutil.rmtree(orig_path)
                
                # Also delete from augmented dataset if it exists
                aug_path = os.path.join(augmented_path, user_to_delete)
                if os.path.exists(aug_path):
                    shutil.rmtree(aug_path)
                    
                print(f"🗑️ Deleted user '{user_to_delete}' from both original and augmented datasets.")
            else:
                print("❌ Deletion canceled.")
        else:
            print("❌ Invalid choice.")
    except ValueError:
        print("❌ Please enter a valid number.")

def train_model():
    print("\n🧠 Starting Face Recognition Model Training...")
    
    # Check if we have users to train on
    users = [d for d in os.listdir(augmented_path) if os.path.isdir(os.path.join(augmented_path, d))]
    if not users:
        print("⚠️ No users found in augmented dataset. Please add users first.")
        return
    
    print(f"Found {len(users)} users for training:")
    for user in users:
        # Count images for this user
        user_dir = os.path.join(augmented_path, user)
        image_count = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')])
        print(f"  - {user}: {image_count} images")
    
    print("\nPreparing training data...")
    
    # Import here to avoid dependencies if not training
    try:
        import pickle
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("❌ Required libraries not found. Please install sklearn.")
        return
    
    # Collect training data
    faces = []
    labels = []
    label_dict = {}
    label_counter = 0
    
    for user in users:
        user_dir = os.path.join(augmented_path, user)
        user_images = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]
        
        if user not in label_dict:
            label_dict[user] = label_counter
            label_counter += 1
        
        print(f"Loading {len(user_images)} images for {user}...")
        for img_file in user_images:
            img_path = os.path.join(user_dir, img_file)
            try:
                face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if face_img is not None:
                    faces.append(face_img)
                    labels.append(label_dict[user])
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    print(f"Total training samples: {len(faces)}")
    
    if len(faces) == 0:
        print("❌ No valid face images found. Training aborted.")
        return
    
    # Create and train the LBPH face recognizer
    print("Training face recognition model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,          # Reduced from default 3
        neighbors=12,      # Increased from default 8
        grid_x=8,          # Default is 8
        grid_y=8,          # Default is 8
        threshold=100.0    # Default is 100.0
    )
    recognizer.train(faces, np.array(labels))
    
    # Save the trained model
    models_dir = os.path.join(os.path.dirname(base_path), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "face_recognizer_model.xml")
    recognizer.write(model_path)
    
    # Save the label mapping
    label_map = {v: k.replace("User_", "") for k, v in label_dict.items()}
    with open(os.path.join(models_dir, "label_map.pkl"), 'wb') as f:
        pickle.dump(label_map, f)
    
    print(f"✅ Training complete! Model saved to {model_path}")
    print(f"✅ Label map saved with {len(label_map)} users")
    
    return True

# === Main loop ===
while True:
    print("\n==========================")
    print("Face Recognition Training System")
    print("==========================")
    print("1. Add User")
    print("2. Delete User")
    print("3. Train Face Recognition Model")
    print("4. Exit")
    choice = input("Select an option (1/2/3/4): ").strip()

    if choice == '1':
        add_user()
    elif choice == '2':
        delete_user()
    elif choice == '3':
        train_model()
    elif choice == '4':
        print("👋 Exiting...")
        break
    else:
        print("❌ Invalid input. Please select 1, 2, 3 or 4.")