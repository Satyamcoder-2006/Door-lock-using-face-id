import os
import cv2
import pickle
import numpy as np
import random
from tqdm import tqdm

def get_user_folders(data_dir):
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("User_")]

def apply_augmentation(img):
    """Apply various augmentations to simulate different lighting conditions"""
    augmented_images = [img]  # Original image is also included

    # Brightness variations
    for alpha in [0.8, 1.2]:  # Darker and brighter
        bright_img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented_images.append(bright_img)

    # Contrast variations
    for alpha in [0.7, 1.3]:  # Lower and higher contrast
        contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=10)
        augmented_images.append(contrast_img)

    # Gaussian blur to simulate focus issues
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    augmented_images.append(blur_img)

    # Shadow simulation (apply a gradient shadow to a random part of the image)
    shadow_img = img.copy()
    rows, cols = shadow_img.shape
    shadow_width = random.randint(cols // 3, cols // 2)
    x1 = random.randint(0, cols - shadow_width)
    x2 = x1 + shadow_width

    shadow_mask = np.ones(shadow_img.shape, dtype=np.float32)
    for i in range(shadow_width):
        shadow_intensity = 0.7 - 0.5 * (i / shadow_width)  # Gradient shadow
        shadow_mask[:, x1 + i] = shadow_intensity

    shadow_img = (shadow_img * shadow_mask).astype(np.uint8)
    augmented_images.append(shadow_img)

    # Noise addition to simulate low light conditions
    noise_img = img.copy()
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noise_img = cv2.add(noise_img, noise)
    augmented_images.append(noise_img)

    return augmented_images

def preprocess_face(img):
    """Apply advanced preprocessing to normalize lighting conditions"""
    if img is None:
        return None

    # Resize if needed
    if img.shape != (200, 200):
        img = cv2.resize(img, (200, 200))

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Normalize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Gamma correction to balance light and dark regions
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    return img

def detect_and_align_face(img, face_detector, face_landmark_detector=None):
    """Detect face and align it based on eyes location if landmark detector is provided"""
    if face_landmark_detector is None:
        # Basic detection without alignment
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None

        (x, y, w, h) = faces[0]  # Use the first face
        return gray[y:y+h, x:x+w]
    else:
        # Implement face alignment using dlib or other landmark detector here
        # This would require additional code for eye alignment
        pass

def load_and_preprocess_images(data_dir, face_detector, valid_extensions, use_augmentation=True):
    faces, labels, label_map = [], [], {}
    label_id = 0

    user_folders = get_user_folders(data_dir)
    if not user_folders:
        raise ValueError("❌ No user folders found in the training data directory.")

    print(f"👥 Found {len(user_folders)} user folders")

    for user_folder in user_folders:
        user_path = os.path.join(data_dir, user_folder)
        image_files = [f for f in os.listdir(user_path) if os.path.splitext(f)[1].lower() in valid_extensions]

        if not image_files:
            print(f"⚠️ Skipping {user_folder} - No valid image files")
            continue

        user_name = user_folder.replace("User_", "")
        label_map[label_id] = user_name
        successful_faces = 0

        print(f"\n🔍 Processing user: {user_name}")

        for img_name in tqdm(image_files, desc=f"Processing {user_name}", leave=False):
            img_path = os.path.join(user_path, img_name)

            # Read original image (in color for better face detection)
            img_color = cv2.imread(img_path)
            if img_color is None:
                print(f"⚠️ Cannot read {img_path}, skipping.")
                continue

            # Convert to grayscale for recognition
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if len(img_color.shape) == 3 else img_color

            try:
                # Detect and align face
                face = detect_and_align_face(img_color, face_detector)
                if face is None:
                    print(f"⚠️ No face detected in {img_path}, skipping.")
                    continue

                # Preprocess the detected face
                face = preprocess_face(face)
                if face is None:
                    continue

                # Add original processed face
                faces.append(face)
                labels.append(label_id)
                successful_faces += 1

                # Apply augmentation if enabled
                if use_augmentation:
                    augmented_faces = apply_augmentation(face)
                    # Skip the first one as it's the original face we already added
                    for aug_face in augmented_faces[1:]:
                        faces.append(aug_face)
                        labels.append(label_id)
                        successful_faces += 1

            except Exception as e:
                print(f"❌ Failed to process image: {img_path}. Reason: {e}")

        if successful_faces == 0:
            print(f"⚠️ No valid faces added for user {user_name}")
            label_map.pop(label_id)
        else:
            label_id += 1
            print(f"✅ {successful_faces} face samples processed for {user_name}")

    if not faces:
        raise ValueError("❌ No valid face images were found for training.")

    return np.array(faces), np.array(labels), label_map

def train_and_save_model(faces, labels, label_map, model_path='face_recognizer_model.xml', map_path='label_map.pkl'):
    print(f"\n✨ Training model with {len(faces)} faces from {len(label_map)} users...")
    try:
        # Create LBPH Face Recognizer with optimized parameters
        model = cv2.face.LBPHFaceRecognizer_create(
            radius=3,      # Increased from 2 for better texture analysis
            neighbors=12,  # Keep at 12 as it works well for varied lighting
            grid_x=9,      # Slightly increased grid size for better spatial info
            grid_y=9,
            threshold=85.0 # Lower threshold for more sensitivity
        )
        model.train(faces, labels)
        model.save(model_path)
        print("✅ Face recognition model saved at:", model_path)

        with open(map_path, 'wb') as f:
            pickle.dump(label_map, f)
        print("✅ Label map saved at:", map_path)

        print("\n🎉 Training complete!")
        print(f"- Total faces: {len(faces)}")
        print(f"- Total users: {len(label_map)}")
        print(f"- Users: {', '.join(label_map.values())}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to train or save the model: {e}")

def train_face_recognizer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'userimage')
    os.makedirs(data_dir, exist_ok=True)

    # Load a more robust face detector
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"❌ Haar cascade not found at {face_cascade_path}")

    face_detector = cv2.CascadeClassifier(face_cascade_path)
    valid_extensions = ['.jpg', '.jpeg', '.png']

    # Process images with augmentation
    faces, labels, label_map = load_and_preprocess_images(
        data_dir,
        face_detector,
        valid_extensions,
        use_augmentation=True
    )

    train_and_save_model(faces, labels, label_map)

if __name__ == '__main__':
    try:
        train_face_recognizer()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
