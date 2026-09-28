# ⚡ Real-Time Face Recognition & Registration System

A fast, real-time facial recognition and identity verification web application built with **Python**, **Flask**, **OpenCV**, **dlib / face_recognition**, and **MySQL**.

---

## 🚀 Features

- **Live Camera Stream**: Real-time video feed with bounding box detection and instantaneous facial identification.
- **In-Memory RAM Caching**: Face encodings are cached in memory to avoid database bottlenecks during high-FPS recognition.
- **Duplicate Prevention**: Rejects new registration attempts if the face is already registered in the system.
- **User Registration Portal**: Capture photo directly from webcam or upload images along with user metadata (Name, Age, Phone, Email, Address, Aadhaar Last 4).
- **Privacy Masking**: Built-in masking for sensitive information such as phone numbers and Aadhaar numbers.
- **Single-Image Recognition**: Upload an image to test recognition independently.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask
- **Computer Vision & ML**: OpenCV (`opencv-python`), `face_recognition`, `dlib`, `numpy`, `Pillow`
- **Database**: MySQL (via `mysql-connector-python`)
- **Frontend**: HTML5, CSS3, JavaScript (Webcam API)

---

## 📂 Project Structure

```text
facedetection/
├── app.py                      # Main Flask application & recognition engine
├── req.txt                     # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── static/
│   ├── style.css               # Styling for web pages
│   ├── captures/               # Captured/registered user images
│   └── uploads/                # Temporary uploaded test images
└── templates/
    ├── final_working.html      # Real-time recognition dashboard
    └── register.html           # User registration page
```

---

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **MySQL Server** (XAMPP / WAMP / MySQL Workbench / standalone service)
3. **C++ Build Tools & CMake** (Required if building `dlib` from source on Windows)

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/anya0123/face-recognition-.git
cd face-recognition-
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r req.txt
```

> **Note for Windows users**: If installing `dlib` fails, install pre-built wheels or `dlib-bin`:
> ```bash
> pip install dlib-bin
> ```

---

## 🗄️ Database Setup

1. Start your **MySQL Server** (e.g., via XAMPP Control Panel).
2. Create a database named `face`:
   ```sql
   CREATE DATABASE face;
   ```
3. *(Optional)* Update database credentials in [`app.py`](file:///c:/Users/anish/OneDrive/Desktop/facedetection/app.py) if needed:
   ```python
   DB_CONFIG = {
       'host': '127.0.0.1',
       'port': 3306,
       'user': 'root',
       'password': '',
       'database': 'face'
   }
   ```
4. The required table (`persons`) will automatically be created on the first run of the app.

---

## ▶️ Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```

---

## 🌐 Routes & Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | `GET` | Main live recognition dashboard |
| `/register` | `GET`, `POST` | User registration interface & API |
| `/recognize` | `POST` | Image upload recognition endpoint |
| `/camera` | `GET` | MJPEG video stream feed |
| `/current_user` | `GET` | Returns currently detected user information in JSON |

---

## 🔒 Security & Best Practices

- Make sure to add `.env` or configuration management for database passwords in production environments.
- Do not commit virtual environments (`venv/`) or sensitive database files.
