import cv2
import os
import shutil

# Load the Haar cascade
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face extraction function
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]

# Base folder for images
base_path = r"C:\Users\satya\OneDrive\Desktop\IOTPROJECTFINAL\userimage"
def add_user():
    user_name = input("Enter new user name: ").strip()
    save_path = os.path.join(base_path, f"User_{user_name}")
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(1)
    count = 0
    max_images = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        face = face_extractor(frame)
        if face is not None:
            count += 1
            face = cv2.resize(face, (200, 200))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name = os.path.join(save_path, f"{user_name}_{count}.jpg")
            cv2.imwrite(file_name, face_gray)

            cv2.putText(face, f"{count}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Collector", face)
        else:
            print("Face not found")
            cv2.imshow("Face Collector", frame)

        if cv2.waitKey(1) == 13 or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collected {count} face samples for User: {user_name}")

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
                shutil.rmtree(os.path.join(base_path, user_to_delete))
                print(f"🗑️ Deleted user '{user_to_delete}' successfully.")
            else:
                print("❌ Deletion canceled.")
        else:
            print("❌ Invalid choice.")
    except ValueError:
        print("❌ Please enter a valid number.")

# === Main loop ===
while True:
    print("\n==========================")
    print("Face Data Manager")
    print("==========================")
    print("1. Add User")
    print("2. Delete User")
    print("3. Exit")
    choice = input("Select an option (1/2/3): ").strip()

    if choice == '1':
        add_user()
    elif choice == '2':
        delete_user()
    elif choice == '3':
        print("👋 Exiting...")
        break
    else:
        print("❌ Invalid input. Please select 1, 2, or 3.")
