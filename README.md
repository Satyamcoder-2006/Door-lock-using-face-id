# 🚪 FaceLock: Smart Face Recognition Door Lock System

![FaceLock Banner](https://img.shields.io/badge/FaceLock-Smart%20Door%20System-blueviolet?style=for-the-badge&logo=python&logoColor=white)

A next-generation smart door lock system using facial recognition, real-time monitoring, and IoT integration. Built with **Python**, **Flask**, **OpenCV**, and **Arduino** for robust security and seamless user experience.

---

## 🌟 Features

- **Face Recognition Authentication**: Only authorized users can unlock the door using advanced face recognition.
- **Admin Dashboard**: Modern, responsive web dashboard for full control and monitoring.
- **User Management**: Add, view, and delete users with live camera capture and face image augmentation.
- **Real-Time Door Control**: Instantly lock/unlock the door via Arduino integration from the dashboard or API.
- **Access Logs & Analytics**: Detailed, filterable logs of all access attempts (granted/denied), with timestamps and user info.
- **Voice Feedback**: Audio prompts for access granted, denied, door status, and personalized greetings.
- **Telegram Notifications**: Get instant alerts for unlocks, locks, and failed access attempts.
- **System Health Monitoring**: Dashboard widgets for camera, face recognition, Arduino, and notification status.
- **Security**: Secure session management, password change, and admin authentication.
- **API Endpoints**: RESTful APIs for door status, lock/unlock, and system integration.
- **Log Management**: Clean, clear, and export logs directly from the dashboard.
- **Dark Mode UI**: Beautiful, modern, and accessible interface with dark mode by default.

---


---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IOTPROJECTFINAL
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
- Set your `SECRET_KEY` in a `.env` file or in `SECRET_KEY=your-secure-secret-key.txt`.
- (Optional) Configure `admin_config.json` and `telegram_config.json` for admin and notifications.

### 4. Connect Hardware
- Connect your Arduino to the system (update the COM port in `app.py` if needed).
- Ensure your camera is connected and accessible.

### 5. Initialize Database (First Time Only)
```bash
python init_db.py
```

### 6. Run the Application
```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🛠️ Main Components

- **app.py**: Main Flask server, routes, and logic
- **database.py**: User and face data management
- **doorlock.py**: Real-time face recognition and Arduino control
- **voice_module.py**: Audio feedback and TTS
- **telegram_notifier.py**: Telegram notification integration
- **templates/**: All HTML templates (dashboard, logs, user management, etc.)
- **static/**: CSS, JS, and audio assets

---

## ⚙️ Features in Detail

### Face Recognition & User Management
- Add users with live camera capture and automatic face image augmentation
- Delete users and retrain the model instantly
- View user images and stats

### Admin Dashboard
- Modern UI with door status, quick actions, system health, and recent activity
- Manual lock/unlock controls
- Access logs with filtering, pagination, and export
- Log cleaning and clearing tools

### Notifications
- Configure Telegram bot and chat IDs for instant alerts
- Enable/disable notifications for unlock, lock, and failed access

### Voice Feedback
- Customizable audio prompts for all major events
- Personalized greetings for recognized users

### Security
- Admin login and password management
- Secure session handling
- Unauthorized access page

### API Endpoints
- `/api/status` — Get door and system status
- `/api/unlock` — Unlock door (POST)
- `/api/lock` — Lock door (POST)

---

## 📦 Dependencies

- Flask==2.0.1
- face-recognition==1.3.0
- numpy==1.21.0
- opencv-python==4.5.3.56
- pyserial==3.5
- python-dotenv==0.19.0
- Pillow==8.3.1
- gTTS, pygame (for voice feedback)

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🤖 Hardware Requirements
- **Arduino** (Uno/Nano/compatible) with relay for door control
- **Webcam** (USB or built-in)
- **Speaker** (for audio feedback)

---

## 🛡️ Security Notes
- Change the default admin password after setup
- Use a strong, unique `SECRET_KEY`
- Keep your Telegram bot token and chat IDs private

---

## 📚 License
MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Credits
- Built by your team
- Powered by [Flask](https://flask.palletsprojects.com/), [OpenCV](https://opencv.org/), [face-recognition](https://github.com/ageitgey/face_recognition), and the open-source community.

---

## 💡 Tips
- For best recognition, capture user faces in good lighting and from multiple angles.
- Regularly check logs and system health from the dashboard.
- Use the Telegram test feature to verify notifications.

---

## 📞 Support
For issues, open an [issue](https://github.com/your-repo/issues) or contact the maintainer.
