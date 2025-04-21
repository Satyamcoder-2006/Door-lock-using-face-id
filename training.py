import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

def get_user_folders(data_dir):
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("User_")]

def load_and_preprocess_images(data_dir, face_classifier, valid_extensions):
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
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Cannot read {img_path}, skipping.")
                continue

            try:
                if img.shape != (200, 200):
                    img = cv2.resize(img, (200, 200))
                img = cv2.equalizeHist(img)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

                faces.append(img)
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
        model = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=12,
            grid_x=8,
            grid_y=8,
            threshold=100.0
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

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"❌ Haar cascade not found at {face_cascade_path}")

    face_classifier = cv2.CascadeClassifier(face_cascade_path)
    valid_extensions = ['.jpg', '.jpeg', '.png']

    faces, labels, label_map = load_and_preprocess_images(data_dir, face_classifier, valid_extensions)
    train_and_save_model(faces, labels, label_map)

if __name__ == '__main__':
    try:
        train_face_recognizer()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
