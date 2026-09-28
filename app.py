import os, json, base64, time, io
import mysql.connector
import numpy as np
import cv2
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from datetime import datetime
from PIL import Image

app = Flask(__name__)
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'face'
}
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

THRESHOLD = 0.48  # recognition tolerance
CAPTURE_FOLDER = "static/captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# In-Memory Cache for ultra-fast recognition without DB bottleneck
CACHED_PEOPLE = []

def init_db():
    try:
        con = mysql.connector.connect(**DB_CONFIG)
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS persons (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            age INT,
            phone VARCHAR(15),
            email VARCHAR(100),
            address TEXT,
            aadhaar_last4 VARCHAR(4),
            encoding TEXT,
            image_path VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        con.commit()
        con.close()
        return True
    except:
        return False

def load_people():
    try:
        con = mysql.connector.connect(**DB_CONFIG)
        cur = con.cursor()
        cur.execute("SELECT name, age, phone, email, address, aadhaar_last4, encoding, image_path FROM persons")
        rows = cur.fetchall()
        con.close()
        people = []
        for name, age, phone, email, address, aadhaar_last4, enc_json, img_path in rows:
            if enc_json:
                enc = np.array(json.loads(enc_json), dtype=np.float64)
                people.append({
                    "name": name, "age": age, "phone": phone, "email": email,
                    "address": address, "aadhaar_last4": aadhaar_last4, "enc": enc, "image_path": img_path
                })
        return people
    except:
        return []

def reload_people_cache():
    global CACHED_PEOPLE
    CACHED_PEOPLE = load_people()
    print(f"⚡ Fast Cache reloaded: {len(CACHED_PEOPLE)} people loaded in RAM.")

def mask_phone(p):
    if not p or p == "N/A":
        return "N/A"
    digits = "".join(ch for ch in p if ch.isdigit())
    tail = digits[-4:] if len(digits) >= 4 else digits
    return f"******{tail}" if tail else "******"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return redirect(url_for('index'))
    return render_template("final_working.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"})

    try:
        # Read file directly in memory for speed
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"success": False, "error": "Invalid image file"})

        # Resize for ultra-fast processing (max 600px width)
        height, width = img.shape[:2]
        if width > 600:
            scale = 600 / width
            new_width = 600
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Fast face detection & single-jitter encoding
        faces = face_recognition.face_locations(rgb_img, model="hog")
        if not faces:
            return jsonify({"success": False, "error": "No face detected"})
        
        encs = face_recognition.face_encodings(rgb_img, [faces[0]], num_jitters=1)
        if not encs:
            return jsonify({"success": False, "error": "Could not encode face"})

        people = CACHED_PEOPLE if CACHED_PEOPLE else load_people()
        if not people:
            return jsonify({"success": True, "name": "Unknown", "image_url": ""})
        
        known_encs = [p["enc"] for p in people]
        enc = encs[0]
        
        # Fast comparison
        dists = face_recognition.face_distance(known_encs, enc)
        best_i = int(np.argmin(dists))
        
        if dists[best_i] < THRESHOLD:
            info = people[best_i]
            return jsonify({
                "success": True,
                "name": info["name"],
                "age": info["age"],
                "phone": mask_phone(info["phone"]),
                "email": info["email"],
                "address": info["address"],
                "aadhaar": "**** **** **** " + info["aadhaar_last4"] if info["aadhaar_last4"] != "N/A" else "N/A",
                "image_url": f"/static/captures/{info['image_path']}" if info['image_path'] else ""
            })
        else:
            return jsonify({"success": True, "name": "Unknown", "image_url": ""})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            data = request.get_json()
            name = data.get('name')
            age = data.get('age')
            phone = data.get('phone')
            email = data.get('email')
            address = data.get('address')
            aadhaar_last4 = data.get('aadhaar_last4')
            image_data = data.get('image')
            
            if not image_data or ',' not in image_data:
                return jsonify({"success": False, "error": "No valid image received"})
            
            # Fast in-memory decoding
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({"success": False, "error": "Could not decode image"})
            
            # Resize image to standard size for ultra-fast processing
            height, width = img.shape[:2]
            if width > 640:
                scale = 640 / width
                img = cv2.resize(img, (640, int(height * scale)), interpolation=cv2.INTER_AREA)
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Fast face detection & encoding
            faces = face_recognition.face_locations(rgb_img, model="hog")
            if not faces:
                return jsonify({"success": False, "error": "No face detected in image"})
            
            encs = face_recognition.face_encodings(rgb_img, [faces[0]], num_jitters=1)
            if not encs:
                return jsonify({"success": False, "error": "Could not encode face"})
            
            # Instant check against cached people in RAM
            people = CACHED_PEOPLE if CACHED_PEOPLE else load_people()
            if people:
                known_encs = [p["enc"] for p in people]
                matches = face_recognition.compare_faces(known_encs, encs[0], tolerance=THRESHOLD)
                if True in matches:
                    match_index = matches.index(True)
                    existing_person = people[match_index]
                    return jsonify({
                        "success": False, 
                        "error": f"Face already registered as {existing_person['name']}",
                        "existing_user": {
                            "name": existing_person["name"],
                            "phone": existing_person["phone"],
                            "email": existing_person["email"]
                        }
                    })
            
            # Save compressed image
            timestamp = str(int(time.time()))
            filename = f"capture_{name}_{timestamp}.jpg"
            image_path = os.path.join(CAPTURE_FOLDER, filename)
            cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            encoding_json = json.dumps(encs[0].tolist())
            
            # Save to database
            con = mysql.connector.connect(**DB_CONFIG)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO persons (name, age, phone, email, address, aadhaar_last4, encoding, image_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (name, age, phone, email, address, aadhaar_last4, encoding_json, filename)
            )
            con.commit()
            con.close()
            
            # Refresh in-memory cache immediately
            reload_people_cache()
            
            return jsonify({"success": True, "message": "User registered successfully!"})
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    return render_template("register.html")

# Real-time camera streaming with high FPS optimization

def gen_frames():
    global current_detected_user
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    face_locations = []
    face_names = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Process every 2nd frame with cached people for smooth real-time tracking
        if frame_count % 2 == 0:
            people = CACHED_PEOPLE
            known_encs = [p["enc"] for p in people] if people else []
            names = [p["name"] for p in people] if people else []
            
            # Resize frame to 1/4 size for ultra-fast HOG detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces & encodings fast
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1) if face_locations else []
            
            face_names = []
            current_frame_users = []
            
            for face_encoding in face_encodings:
                name = "Unknown"
                if known_encs:
                    dists = face_recognition.face_distance(known_encs, face_encoding)
                    best_match = int(np.argmin(dists))
                    if dists[best_match] < THRESHOLD:
                        name = names[best_match]
                        person = people[best_match]
                        user_data = {
                            "name": person["name"],
                            "age": person["age"],
                            "phone": person["phone"] or "N/A",
                            "email": person["email"],
                            "address": person["address"],
                            "aadhaar": "**** **** **** " + person["aadhaar_last4"] if person["aadhaar_last4"] != "N/A" else "N/A",
                            "image_path": person["image_path"],
                            "status": "VERIFIED"
                        }
                        if not any(u["name"] == user_data["name"] for u in current_frame_users):
                            current_frame_users.append(user_data)
                face_names.append(name)
            
            if current_frame_users:
                current_detected_user = current_frame_users
        
        # Draw bounding boxes and tags
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 32), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        frame_count += 1
        
        # Fast JPEG encoding (Quality 72 for max speed)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

current_detected_user = None

@app.route('/camera')
def camera():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_user')
def get_current_user():
    global current_detected_user
    if current_detected_user:
        return jsonify({
            "success": True,
            "users": current_detected_user,
            "count": len(current_detected_user)
        })
    else:
        return jsonify({
            "success": False,
            "message": "No user detected"
        })

@app.route('/search_users')
def search_users():
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify({"success": False, "error": "Search query is required"})
    
    # Fast in-memory search over cached users
    people = CACHED_PEOPLE if CACHED_PEOPLE else load_people()
    matched = []
    for p in people:
        if (query in (p['name'] or '').lower() or 
            query in (p['phone'] or '').lower() or 
            query in (p['email'] or '').lower() or 
            query in (p['address'] or '').lower()):
            matched.append({
                "name": p['name'],
                "age": p['age'],
                "phone": mask_phone(p['phone']),
                "email": p['email'],
                "address": p['address'],
                "aadhaar": "**** **** **** " + p['aadhaar_last4'] if p['aadhaar_last4'] != "N/A" else "N/A",
                "image_path": p['image_path']
            })
    return jsonify({"success": True, "users": matched, "count": len(matched)})

@app.route('/check_face', methods=['POST'])
def check_face():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data or ',' not in image_data:
            return jsonify({"success": False, "error": "No image provided"})
        
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"success": False, "error": "Invalid image"})
        
        height, width = img.shape[:2]
        if width > 500:
            scale = 500 / width
            img = cv2.resize(img, (500, int(height * scale)), interpolation=cv2.INTER_AREA)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_img, model="hog")
        if not faces:
            return jsonify({"success": False, "error": "No face detected"})
        
        encs = face_recognition.face_encodings(rgb_img, [faces[0]], num_jitters=1)
        if not encs:
            return jsonify({"success": False, "error": "Could not encode face"})
        
        people = CACHED_PEOPLE if CACHED_PEOPLE else load_people()
        if people:
            known_encs = [p["enc"] for p in people]
            distances = face_recognition.face_distance(known_encs, encs[0])
            min_idx = np.argmin(distances)
            if distances[min_idx] < THRESHOLD:
                existing_person = people[min_idx]
                return jsonify({
                    "success": True,
                    "face_exists": True,
                    "existing_user": {
                        "name": existing_person["name"],
                        "phone": existing_person["phone"],
                        "email": existing_person["email"]
                    }
                })
        
        return jsonify({"success": True, "face_exists": False})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    if init_db():
        print("Database initialized successfully")
    else:
        print("Warning: Could not initialize database")
    
    reload_people_cache()
    
    print("⚡ High-Speed Face Recognition System is starting...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
