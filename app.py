# app.py - Main Flask application
import os
import cv2
import pickle
import threading
import time
import shutil
import subprocess
import sys
import json
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, session
from utils import ArduinoManager, trigger_unlock, enhance_image, find_arduino_port, try_all_available_ports
from serial.tools import list_ports
from admin_auth import verify_password, change_password
from access_log import log_access, get_logs
from clear_logs import clear_logs_directly
from telegram_notifier import get_notifier
door_status = "Locked"
last_recognition_time = 0
door_open_duration = 10  # seconds
last_door_action = None


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key in production

# No setup needed for clear logs routes

# Debug available COM ports
print("Available COM ports:")
for port in list_ports.comports():
    print(f"- {port.device}: {port.description}")

# Configuration
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "userimage")
arduino_port = 'COM7'
arduino_baudrate = 9600
camera_index = 1  # Camera is on index 1

# Global variables
camera = None
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
door_status = "Locked"
last_face_check = 0
face_check_interval = 0.5  # seconds

# Initialize Arduino manager
arduino_manager = None
try:
    arduino_manager = ArduinoManager(arduino_port, arduino_baudrate)
    print(f"🔌 Arduino manager initialized with port {arduino_port}")
    # Test the connection by sending a test command
    arduino_manager.write(b't')  # test command
    arduino_manager.flush()
    time.sleep(0.5)
    response = arduino_manager.read(size=10)
    if response:
        print(f"✅ Arduino responded: {response}")
    else:
        print("⚠️ No response from Arduino, but connection established")
except Exception as e:
    print(f"Error initializing Arduino manager: {e}")
    available_ports = find_arduino_port()
    if available_ports:
        print(f"Trying alternative ports: {available_ports}")
        for port in available_ports:
            try:
                arduino_manager = ArduinoManager(port, arduino_baudrate)
                arduino_manager.write(b't')
                arduino_manager.flush()
                time.sleep(0.5)
                response = arduino_manager.read(size=10)
                if response:
                    print(f"✅ Arduino responded on port {port}: {response}")
                    arduino_port = port
                    break
            except Exception as e:
                print(f"Failed on port {port}: {e}")
                continue

# Function to send command to Arduino with verification
def send_arduino_command(command, retries=3):
    global arduino_manager
    if not arduino_manager:
        print("❌ Arduino manager not initialized")
        try:
            print("🔄 Attempting to reinitialize Arduino manager...")
            arduino_manager = ArduinoManager(arduino_port, arduino_baudrate)
            if not arduino_manager.ser:
                print("❌ Failed to reinitialize Arduino manager")
                return False
            print("✅ Arduino manager reinitialized successfully")
        except Exception as e:
            print(f"❌ Error reinitializing Arduino manager: {e}")
            return False

    cmd_type = "unlock" if command == b'u' else "lock" if command == b'l' else "other"
    print(f"🔌 Sending {cmd_type} command to Arduino...")

    for attempt in range(retries):
        try:
            # Send command multiple times to ensure receipt
            for _ in range(3):
                arduino_manager.write(command)
                arduino_manager.flush()
                time.sleep(0.1)

            # Check for response
            response = arduino_manager.read(size=10)
            if response:
                print(f"✅ Arduino acknowledged {cmd_type} command: {response}")
                return True
            print(f"⚠️ No response from Arduino for {cmd_type} command, attempt {attempt + 1}/{retries}")

            # If no response, try again with a different approach
            if attempt == retries - 1 and cmd_type == "lock":
                print("🔄 Trying alternative lock command format...")
                arduino_manager.write(b'LOCK\n')
                arduino_manager.flush()
                time.sleep(0.2)
                response = arduino_manager.read(size=10)
                if response:
                    print(f"✅ Arduino acknowledged alternative lock command: {response}")
                    return True
        except Exception as e:
            print(f"⚠️ Arduino {cmd_type} command failed: {e}")
            try:
                print("🔄 Attempting to reconnect to Arduino...")
                arduino_manager = ArduinoManager(arduino_port, arduino_baudrate)
                if not arduino_manager.ser:
                    print("❌ Failed to reconnect to Arduino")
            except Exception as reconnect_error:
                print(f"❌ Error reconnecting to Arduino: {reconnect_error}")

    # If we get here, all attempts failed
    print(f"❌ All attempts to send {cmd_type} command failed")
    return False

# Load face recognizer model and label map (only once!)
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists('face_recognizer_model.xml') or not os.path.exists('label_map.pkl'):
            print("⚠️ Model files not found. Checking for users...")
            # Check if we have any users
            users = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
            if users:
                print(f"⚠️ Found {len(users)} users but no model. Attempting to train...")
                try:
                    from training import train_face_recognizer
                    train_face_recognizer()
                    print("✅ Successfully trained new model")
                except Exception as train_error:
                    print(f"❌ Failed to train model: {train_error}")
                    return None, {}
            else:
                print("ℹ️ No users found. No model needed yet.")
                return None, {}

        # Now try to load the model
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read('face_recognizer_model.xml')

        with open('label_map.pkl', 'rb') as f:
            label_map = pickle.load(f)

        print(f"✅ Successfully loaded model with {len(label_map)} users")
        return model, label_map
    except Exception as e:
        print(f"❌ Error loading face recognition model: {e}")
        return None, {}

model, label_map = load_model()

# Unlock helper
def ensure_door_unlock():
    print("🔓 Attempting direct door unlock")
    try:
        if trigger_unlock():
            print("✅ Door unlocked using trigger_unlock")
            return True
    except Exception as e:
        print(f"⚠️ Primary unlock failed: {e}")
    try:
        with ArduinoManager(arduino_port, arduino_baudrate) as arduino:
            if arduino:
                signals = [b'u', b'UNLOCK\n', b'unlock\n']
                for signal in signals:
                    for _ in range(3):
                        arduino.write(signal)
                        arduino.flush()
                        time.sleep(0.2)
                        response = arduino.read(size=10)
                        if response:
                            print(f"✅ Door unlock confirmed: {response}")
                            return True
                print("Sent unlock signals (no confirmation)")
                return True
    except Exception as e:
        print(f"⚠️ Arduino fallback failed: {e}")
    try:
        arduino = try_all_available_ports()
        if arduino:
            arduino.write(b'UNLOCK\n')
            arduino.flush()
            time.sleep(0.5)
            print("✅ Door unlock via fallback port")
            return True
    except Exception as e:
        print(f"⚠️ Unlock on alternative ports failed: {e}")
    return False

# Streaming and recognition
def generate_frames():
    global camera, door_status, last_face_check, model, label_map
    global last_recognition_time, door_open_duration, last_door_action

    try:
        if camera is None or not camera.isOpened():
            print("📸 Initializing camera...")
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                print("❌ Failed to open camera")
                return
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 15)
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        while True:
            success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Handle case when no model is loaded or no users are registered
            if model is None or not label_map:
                door_status = "Locked"
                if last_door_action != "lock":
                    if send_arduino_command(b'l'):
                        last_door_action = "lock"

                # Display message on frame
                cv2.putText(frame, "No users registered", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Add users via Admin panel", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            status_color = (0, 255, 0) if door_status == "Unlocked" else (0, 0, 255)
            cv2.putText(frame, f"Door: {door_status}", (frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            current_time = time.time()

            # Check if model and label_map are valid
            if model is None or not label_map:
                cv2.putText(frame, "No users registered", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            if (current_time - last_face_check) >= face_check_interval:
                last_face_check = current_time
                enhanced = enhance_image(frame)
                faces = face_classifier.detectMultiScale(
                    enhanced,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                if len(faces) > 0:
                    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face = enhanced[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))

                    try:
                        label, confidence = model.predict(face)
                        name = label_map.get(label, "Unknown")

                        if confidence < 105 and name != "Unknown":
                            cv2.putText(frame, f"Welcome {name}", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            last_recognition_time = current_time

                            # 🔓 Only send unlock if it's not already sent
                            if door_status != "Unlocked" or last_door_action != "unlock":
                                if send_arduino_command(b'u'):
                                    door_status = "Unlocked"
                                    last_door_action = "unlock"
                                    print("✅ Door unlocked successfully")
                                    # Log successful access
                                    log_access(name, True, "Face Recognition")
                                    # Send Telegram notification
                                    try:
                                        notifier = get_notifier()
                                        notifier.send_door_unlock_notification(name, "Face Recognition")
                                    except Exception as e:
                                        print(f"⚠️ Telegram notification error: {e}")
                        else:
                            cv2.putText(frame, "Access Denied", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            # Log failed access attempt
                            log_access(name if name != "Unknown" else "Unknown Person", False, "Face Recognition")
                    except Exception as e:
                        cv2.putText(frame, "Recognition Error", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(f"⚠️ Face recognition error: {e}")

            # 🔒 Auto-lock if user has left for 10 seconds
            if door_status == "Unlocked" and (time.time() - last_recognition_time > door_open_duration):
                if last_door_action != "lock":
                    if send_arduino_command(b'l'):
                        door_status = "Locked"
                        last_door_action = "lock"
                        print("🔒 Door locked successfully")
                        # Send Telegram notification for auto-lock
                        try:
                            notifier = get_notifier()
                            notifier.send_door_lock_notification()
                        except Exception as e:
                            print(f"⚠️ Telegram notification error: {e}")

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        print(f"❌ generate_frames error: {str(e)}")
        if camera:
            camera.release()
        camera = None

# Routes
@app.route('/')
def index():
    users = []
    try:
        with open("label_map.pkl", "rb") as f:
            label_map = pickle.load(f)
            users = [{'name': name} for name in label_map.values()]
    except Exception as e:
        print(f"Error loading users: {e}")
    return render_template('index.html', users=users, door_status=door_status, session=session)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_user')
def add_user_form():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next=url_for('add_user_form')))
    return render_template('add_user.html')

@app.route('/add_user_process', methods=['POST'])
def add_user_process():
    global camera, model, label_map
    try:
        user_name = request.form.get('username').strip()
        if not user_name:
            return jsonify({'success': False, 'message': 'Username is required'}), 400

        save_path = os.path.join(base_path, f"User_{user_name}")
        if os.path.exists(save_path):
            return jsonify({'success': False, 'message': 'User already exists'}), 400

        os.makedirs(save_path, exist_ok=True)

        if camera is None:
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                return jsonify({'success': False, 'message': 'Failed to initialize camera'}), 500

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 15)
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        count, max_images, failed = 0, 100, 0
        while count < max_images and failed < 300:
            success, frame = camera.read()
            if not success:
                failed += 1
                continue

            enhanced = enhance_image(frame)
            faces = face_classifier.detectMultiScale(enhanced, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

            if len(faces) == 0:
                failed += 1
                continue

            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            if w < 100 or h < 100:
                failed += 1
                continue

            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            face_gray = cv2.equalizeHist(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))

            file_name = os.path.join(save_path, f"{user_name}_{count+1}.jpg")
            if cv2.imwrite(file_name, face_gray):
                count += 1

        if count == 0:
            shutil.rmtree(save_path)
            return jsonify({'success': False, 'message': 'Failed to capture any valid face images'}), 400

        # Train the model
        try:
            from training import train_face_recognizer
            train_face_recognizer()
            # Explicitly reload the model after training
            model, label_map = load_model()
            if model is None or not label_map:
                raise Exception("Failed to reload model after training")
            return jsonify({
                'success': True,
                'message': f"Successfully added user '{user_name}' with {count} face samples and model retrained",
                'count': count
            })
        except Exception as e:
            shutil.rmtree(save_path)
            return jsonify({'success': False, 'message': f'Failed to train model: {str(e)}'}), 500

    except Exception as e:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/delete_user', methods=['GET'])
def list_users_for_deletion():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next=url_for('list_users_for_deletion')))

    users = []
    try:
        with open("label_map.pkl", "rb") as f:
            label_map = pickle.load(f)
            users = [{'name': name} for name in label_map.values()]
    except Exception as e:
        print(f"Error loading users: {e}")
    return render_template('delete_user.html', users=users)

@app.route('/delete_user/<username>', methods=['GET', 'POST'])
def delete_user(username):
    # Strict check for admin authentication
    if not session.get('admin_logged_in'):
        if request.method == 'GET':
            return redirect(url_for('admin_login', next=url_for('list_users_for_deletion')))
        else:
            return jsonify({'success': False, 'message': 'Unauthorized. Please login as admin.'}), 401

    global model, label_map
    try:
        folder_path = os.path.join(base_path, f"User_{username}")

        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # POST request - perform deletion
        if request.method == 'POST':
            try:
                # Lock the door immediately for security
                global door_status, last_door_action
                if door_status == "Unlocked":
                    print(f"🔒 Locking door due to user deletion...")
                    if send_arduino_command(b'l'):
                        door_status = "Locked"
                        last_door_action = "lock"
                        print(f"🔒 Door locked successfully after user deletion request")

                # Create backup of model files before deletion
                backup_dir = "model_backups"
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                if os.path.exists('face_recognizer_model.xml'):
                    backup_model_path = os.path.join(backup_dir, f"face_recognizer_model_{timestamp}.xml")
                    shutil.copy2('face_recognizer_model.xml', backup_model_path)
                    print(f"📂 Backed up model to {backup_model_path}")

                if os.path.exists('label_map.pkl'):
                    backup_label_path = os.path.join(backup_dir, f"label_map_{timestamp}.pkl")
                    shutil.copy2('label_map.pkl', backup_label_path)
                    print(f"📂 Backed up label map to {backup_label_path}")

                # Delete the user folder
                print(f"🗑️ Deleting user folder: {folder_path}")
                shutil.rmtree(folder_path)

                # Check if there are any remaining users
                remaining_users = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

                if not remaining_users:
                    # If no users left, delete the model files
                    if os.path.exists('face_recognizer_model.xml'):
                        os.remove('face_recognizer_model.xml')
                    if os.path.exists('label_map.pkl'):
                        os.remove('label_map.pkl')
                    model, label_map = None, {}
                    return jsonify({
                        'success': True,
                        'message': f'User {username} deleted successfully. No users remaining.'
                    })

                # Train the model after deletion if there are remaining users
                try:
                    print(f"🔄 Retraining model after deleting user {username}...")
                    from training import train_face_recognizer
                    train_face_recognizer()

                    # Reload the model after training
                    print("📂 Loading newly trained model...")
                    model, label_map = load_model()

                    if model is None or not label_map:
                        raise Exception("Failed to load model after retraining")

                    print(f"✅ User {username} deleted and model retrained successfully")
                    return jsonify({
                        'success': True,
                        'message': f'User {username} deleted successfully and model retrained'
                    })
                except Exception as e:
                    print(f"❌ Training error after deletion: {str(e)}")
                    # If training fails, try to reload the model anyway
                    try:
                        # Try to load the existing model first
                        model, label_map = load_model()
                        if model is not None and label_map:
                            print("⚠️ Managed to reload existing model")
                            return jsonify({
                                'success': True,
                                'message': f'User {username} deleted but model retraining had issues. System is still operational.'
                            })

                        # If that fails, try to restore from backup
                        print("🔄 Attempting to restore from backup...")
                        backup_dir = "model_backups"
                        if os.path.exists(backup_dir):
                            # Find the most recent backup files
                            model_backups = [f for f in os.listdir(backup_dir) if f.startswith("face_recognizer_model_")]
                            label_backups = [f for f in os.listdir(backup_dir) if f.startswith("label_map_")]

                            if model_backups and label_backups:
                                # Sort by timestamp (newest first)
                                model_backups.sort(reverse=True)
                                label_backups.sort(reverse=True)

                                # Restore the most recent backups
                                latest_model = os.path.join(backup_dir, model_backups[0])
                                latest_label = os.path.join(backup_dir, label_backups[0])

                                shutil.copy2(latest_model, 'face_recognizer_model.xml')
                                shutil.copy2(latest_label, 'label_map.pkl')

                                print(f"✅ Restored from backup: {model_backups[0]} and {label_backups[0]}")

                                # Try to load the restored model
                                model, label_map = load_model()
                                if model is not None and label_map:
                                    print("✅ Successfully loaded backup model")
                                    return jsonify({
                                        'success': True,
                                        'message': f'User {username} deleted. Model restored from backup.'
                                    })

                        print("❌ Failed to load any model or restore from backup")
                    except Exception as load_error:
                        print(f"❌ Error during recovery: {str(load_error)}")

                    return jsonify({
                        'success': False,
                        'message': f'User deleted but failed to retrain model: {str(e)}. Please restart the system.'
                    }), 500

            except Exception as e:
                print(f"Error during deletion: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': f'Error deleting user: {str(e)}'
                }), 500

        # GET request - return confirmation
        return jsonify({
            'success': True,
            'message': f'Are you sure you want to delete user {username}?',
            'username': username
        })

    except Exception as e:
        print(f"General error in delete_user: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        }), 500

@app.route('/unlock_door')
def manual_unlock():
    global door_status
    if ensure_door_unlock():
        door_status = "Unlocked"
        # Log emergency unlock
        user = "Admin" if session.get('admin_logged_in') else "Emergency User"
        log_access(user, True, "Emergency Unlock")
        threading.Thread(target=lambda: (time.sleep(3), setattr(globals(), 'door_status', 'Locked'))).start()
        return jsonify({"status": "success", "message": "Door unlocked"})
    else:
        # Log failed emergency unlock
        user = "Admin" if session.get('admin_logged_in') else "Emergency User"
        log_access(user, False, "Emergency Unlock (Failed)")
        return jsonify({"status": "error", "message": "Unlock failed"})

@app.route('/unauthorized')
def unauthorized():
    return render_template('unauthorized.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        if verify_password(password):
            session['admin_logged_in'] = True
            next_url = request.form.get('next') or '/admin/dashboard'
            return redirect(next_url)
        return render_template('admin_login.html', error='Invalid password')
    return render_template('admin_login.html')

@app.route('/admin/change-password', methods=['GET', 'POST'])
@app.route('/change_password', methods=['GET', 'POST'])  # Add an alias route for the dashboard link
def change_password_route():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            return render_template('change_password.html', error='New passwords do not match')

        if change_password(current_password, new_password):
            return render_template('change_password.html', success='Password changed successfully')
        else:
            return render_template('change_password.html', error='Current password is incorrect')

    return render_template('change_password.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next='/admin/dashboard'))

    # Format last recognition time for display
    last_seen = None
    if last_recognition_time:
        elapsed = time.time() - last_recognition_time
        if elapsed < 60:
            last_seen = f"{int(elapsed)} seconds ago"
        elif elapsed < 3600:
            last_seen = f"{int(elapsed/60)} minutes ago"
        else:
            last_seen = f"{int(elapsed/3600)} hours ago"

    try:
        return render_template('admin_dashboard.html',
                           door_status=door_status,
                           last_recognition_time=last_seen)
    except Exception as e:
        print(f"Error rendering admin dashboard: {e}")
        return redirect('/')

@app.route('/admin/access_logs')
def access_logs():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next=url_for('access_logs')))

    # Get query parameters for filtering
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    user_filter = request.args.get('user', '')
    access_filter = request.args.get('access')

    # Convert access_filter to boolean if provided
    if access_filter is not None:
        if access_filter.lower() == 'true':
            access_filter = True
        elif access_filter.lower() == 'false':
            access_filter = False
        else:
            access_filter = None

    # Calculate offset for pagination
    offset = (page - 1) * limit

    # Get logs with filtering and pagination
    logs, total_count = get_logs(limit=limit, offset=offset,
                                user_filter=user_filter,
                                access_filter=access_filter)

    # Calculate total pages
    total_pages = (total_count + limit - 1) // limit

    return render_template('access_logs.html',
                          logs=logs,
                          page=page,
                          total_pages=total_pages,
                          user_filter=user_filter,
                          access_filter=access_filter,
                          limit=limit,
                          total_count=total_count)

@app.route('/clear_logs_page')
def clear_logs_page():
    """Show the clear logs page."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('clear_logs.html')

@app.route('/direct_clear')
def direct_clear():
    """Direct access to clear logs page."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('direct_clear.html')

@app.route('/direct_clear_action', methods=['POST'])
def direct_clear_action():
    """Direct action to clear all logs."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        # Use the clear_logs_directly function from access_log.py
        from access_log import clear_logs_directly
        success = clear_logs_directly()

        if success:
            # Log this action
            log_access("Admin", True, "Cleared Access Logs (Direct Action)")
            print("Logs cleared successfully via direct action")
        else:
            print("Failed to clear logs via direct action")

        # Redirect back to access logs page
        return redirect(url_for('access_logs'))
    except Exception as e:
        print(f"Error clearing logs: {str(e)}")
        return redirect(url_for('access_logs'))

@app.route('/clear_logs_action', methods=['POST'])
def clear_logs_action():
    """Action to clear all logs."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        # Try multiple methods to clear logs
        print("Action route: Attempting to clear logs...")

        # Method 1: Use the clear_logs_directly function
        try:
            clear_logs_directly()
            print("Method 1 succeeded")
        except Exception as e1:
            print(f"Method 1 failed: {e1}")

        # Method 2: Run the command-line script
        try:
            import subprocess
            result = subprocess.run([sys.executable, 'clear_logs_cmd.py'],
                                   capture_output=True, text=True)
            print(f"Method 2 output: {result.stdout}")
            if result.returncode != 0:
                print(f"Method 2 error: {result.stderr}")
            else:
                print("Method 2 succeeded")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")

        # Method 3: Direct file write
        try:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs.json')
            with open(log_file, 'w') as f:
                json.dump([], f)
            print(f"Method 3 succeeded: Directly wrote to {log_file}")
        except Exception as e3:
            print(f"Method 3 failed: {e3}")

        print("Action route: Logs cleared successfully using multiple methods")

        # Log this action
        log_access("Admin", True, "Cleared Access Logs (Action Route)")

        # Redirect back to access logs page
        return redirect(url_for('access_logs'))
    except Exception as e:
        print(f"Action route: Error clearing logs: {str(e)}")
        return redirect(url_for('access_logs'))
# Route to manually unlock the door (e.g. via dashboard button)
@app.route('/unlockdoor', methods=['POST'])
def unlockdoor():
    global door_status, last_door_action, last_recognition_time
    try:
        if send_arduino_command(b'u'):  # Send command to unlock door
            door_status = "Unlocked"
            last_door_action = "unlock"
            last_recognition_time = time.time()  # Reset recognition time on manual unlock
            # Log manual unlock
            user = "Admin" if session.get('admin_logged_in') else "Unknown User"
            log_access(user, True, "Manual Unlock")
            # Send Telegram notification
            try:
                notifier = get_notifier()
                notifier.send_door_unlock_notification(user, "Manual Unlock")
            except Exception as e:
                print(f"⚠️ Telegram notification error: {e}")
            return jsonify({'success': True, 'message': 'Door unlocked'})
        else:
            # Log failed manual unlock attempt
            user = "Admin" if session.get('admin_logged_in') else "Unknown User"
            log_access(user, False, "Manual Unlock (Failed)")
            return jsonify({'success': False, 'message': 'Failed to unlock door'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Route to get the current door status (e.g. for dashboard status indicator)
@app.route('/getdoorstatus')
def getdoorstatus():
    return jsonify({'door_status': door_status})

# Route to get the time since last successful face recognition
@app.route('/getlastseen')
def getlastseen():
    if last_recognition_time:
        elapsed = time.time() - last_recognition_time
        return jsonify({'last_seen_seconds_ago': int(elapsed)})
    else:
        return jsonify({'last_seen_seconds_ago': None})

# API route for admin dashboard status
@app.route('/api/status')
def api_status():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    last_seen = None
    if last_recognition_time:
        elapsed = time.time() - last_recognition_time
        if elapsed < 60:
            last_seen = f"{int(elapsed)} seconds ago"
        elif elapsed < 3600:
            last_seen = f"{int(elapsed/60)} minutes ago"
        else:
            last_seen = f"{int(elapsed/3600)} hours ago"

    return jsonify({
        'door_status': door_status,
        'door_status_class': 'text-success' if door_status == 'Unlocked' else 'text-danger',
        'last_recognition_time': last_seen
    })

# API route for admin dashboard unlock
@app.route('/api/unlock', methods=['POST'])
def api_unlock():
    global door_status, last_door_action, last_recognition_time
    if not session.get('admin_logged_in'):
        # Log unauthorized API unlock attempt
        log_access("Unauthorized User", False, "API Unlock (Unauthorized)")
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        if send_arduino_command(b'u'):
            door_status = "Unlocked"
            last_door_action = "unlock"
            last_recognition_time = time.time()
            # Log successful API unlock
            log_access("Admin", True, "API Unlock")
            # Send Telegram notification
            try:
                notifier = get_notifier()
                notifier.send_door_unlock_notification("Admin", "API Unlock")
            except Exception as e:
                print(f"⚠️ Telegram notification error: {e}")
            return jsonify({'success': True, 'message': 'Door unlocked successfully'})
        else:
            # Log failed API unlock
            log_access("Admin", False, "API Unlock (Failed)")
            return jsonify({'success': False, 'message': 'Failed to unlock door'})
    except Exception as e:
        # Log error in API unlock
        log_access("Admin", False, f"API Unlock (Error: {str(e)})")
        return jsonify({'success': False, 'message': str(e)}), 500

# API route for admin dashboard lock
@app.route('/api/lock', methods=['POST'])
def api_lock():
    global door_status, last_door_action
    print("🔒 API lock endpoint called")

    if not session.get('admin_logged_in'):
        print("❌ Unauthorized lock attempt")
        # Log unauthorized API lock attempt
        log_access("Unauthorized User", False, "API Lock (Unauthorized)")
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        print("🔒 Sending lock command to Arduino...")
        result = send_arduino_command(b'l')
        print(f"🔒 Arduino command result: {result}")

        if result:
            door_status = "Locked"
            last_door_action = "lock"
            # Log successful API lock
            log_access("Admin", True, "API Lock")
            print("✅ Door locked successfully")
            # Send Telegram notification
            try:
                notifier = get_notifier()
                notifier.send_door_lock_notification("Admin")
            except Exception as e:
                print(f"⚠️ Telegram notification error: {e}")
            return jsonify({'success': True, 'message': 'Door locked successfully'})
        else:
            # Log failed API lock
            log_access("Admin", False, "API Lock (Failed)")
            print("❌ Failed to lock door")
            return jsonify({'success': False, 'message': 'Failed to lock door'})
    except Exception as e:
        # Log error in API lock
        print(f"❌ Error in API lock: {str(e)}")
        log_access("Admin", False, f"API Lock (Error: {str(e)})")
        return jsonify({'success': False, 'message': str(e)}), 500



@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

@app.route('/test_api', methods=['GET', 'POST'])
def test_api():
    """Test route to verify API functionality."""
    return jsonify({
        'success': True,
        'message': 'API is working correctly',
        'method': request.method,
        'admin_logged_in': session.get('admin_logged_in', False)
    })

@app.route('/debug_routes')
def debug_routes():
    """Debug route to list all available routes."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify(routes)

@app.route('/admin/telegram_config', methods=['GET', 'POST'])
def telegram_config():
    """Configure Telegram notifications."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next='/admin/telegram_config'))

    notifier = get_notifier()
    success_message = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get form data
            enabled = 'enabled' in request.form
            bot_token = request.form.get('bot_token', '').strip()
            chat_ids_str = request.form.get('chat_ids', '').strip()
            chat_ids = [cid.strip() for cid in chat_ids_str.split(',') if cid.strip()]

            # Get notification types
            door_unlock = 'door_unlock' in request.form
            door_lock = 'door_lock' in request.form
            failed_access = 'failed_access' in request.form

            # Update config
            config = notifier.config.copy()
            config['enabled'] = enabled
            config['bot_token'] = bot_token
            config['chat_ids'] = chat_ids
            config['notification_types'] = {
                'door_unlock': door_unlock,
                'door_lock': door_lock,
                'failed_access': failed_access
            }

            # Save config
            if notifier.save_config(config):
                success_message = "Telegram notification settings saved successfully."
            else:
                error_message = "Failed to save Telegram notification settings."
        except Exception as e:
            error_message = f"Error saving Telegram configuration: {str(e)}"

    return render_template('telegram_config.html',
                           config=notifier.config,
                           success_message=success_message,
                           error_message=error_message)

@app.route('/admin/test_telegram')
def test_telegram():
    """Send a test Telegram notification."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next='/admin/telegram_config'))

    notifier = get_notifier()
    success = notifier.test_notification()

    if success:
        return redirect('/admin/telegram_config?success_message=Test message sent successfully!')
    else:
        return redirect('/admin/telegram_config?error_message=Failed to send test message. Please check your configuration.')

@app.route('/test_lock')
def test_lock():
    """Test route to verify lock functionality."""
    global door_status, last_door_action
    print("🔒 Testing lock functionality...")

    try:
        # Try different lock commands
        commands = [b'l', b'LOCK\n', b'lock\n']
        results = []

        for cmd in commands:
            print(f"🔒 Testing lock command: {cmd}")
            result = send_arduino_command(cmd)
            results.append({
                'command': str(cmd),
                'success': result
            })
            if result:
                door_status = "Locked"
                last_door_action = "lock"
                print("✅ Lock command succeeded")
                break
            else:
                print("❌ Lock command failed")

        return jsonify({
            'success': any(r['success'] for r in results),
            'door_status': door_status,
            'test_results': results
        })
    except Exception as e:
        print(f"❌ Error in test_lock: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/run_batch_file')
def run_batch_file():
    """Run the batch file to clear logs."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    try:
        # Run the batch file
        print("Running batch file to clear logs...")
        batch_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clear_logs.bat')
        subprocess.Popen(batch_file, shell=True)

        # Log this action
        log_access("Admin", True, "Cleared Access Logs (Batch File)")

        # Redirect back to access logs page
        return redirect(url_for('access_logs'))
    except Exception as e:
        print(f"Error running batch file: {str(e)}")
        return redirect(url_for('access_logs'))

if __name__ == '__main__':
    try:
        # Ensure user image directory exists
        os.makedirs(base_path, exist_ok=True)

        # Create model_backups directory if it doesn't exist
        os.makedirs("model_backups", exist_ok=True)

        # Load the model (the enhanced load_model function will handle missing files)
        model, label_map = load_model()

        # Print system status
        users = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        print(f"ℹ️ System Status: {len(users)} users registered")
        if model is not None and label_map:
            print("✅ Face recognition model loaded successfully")
        else:
            print("ℹ️ No face recognition model loaded. System will start without recognition capabilities.")
            print("ℹ️ Add users through the admin interface to enable face recognition.")

        # Start the Flask application
        print("🚀 Starting Face Recognition Door Lock System...")
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ App startup error: {e}")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
