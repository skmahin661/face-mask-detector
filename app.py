from flask import Flask, render_template, request, Response
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# === Model Download ===
model_path = "maask_detector.h5"
gdrive_file_id = "19rqkvxGHGIDMzyZ3AB0MyCVxyCdwBLCQ"
url = f"https://drive.google.com/uc?id={gdrive_file_id}"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# === Load Model and Haar Cascade ===
model = load_model(model_path)
labels = ['No Mask', 'Mask', 'Incorrect']
colors = [(0, 0, 255), (0, 255, 0), (0, 165, 255)]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# === Image Upload Prediction ===
def detect_faces_in_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.astype("float") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = model.predict(face_img)[0]
        label_index = np.argmax(preds)
        label = labels[label_index]
        color = colors[label_index]

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label} ({int(preds[label_index]*100)}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
    cv2.imwrite(output_path, img)
    return "output.jpg"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output_filename = detect_faces_in_image(file_path)
            return render_template('result.html', output_image=output_filename)

    return render_template('index.html')


# === Live Camera Feed Prediction ===
def generate_camera_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype("float") / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            preds = model.predict(face_img)[0]
            label_index = np.argmax(preds)
            label = labels[label_index]
            color = colors[label_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({int(preds[label_index]*100)}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/live')
def live():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# === Run App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
