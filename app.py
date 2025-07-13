from flask import Flask, render_template, Response, redirect, url_for
import cv2
from deepface import DeepFace
import threading
import os

app = Flask(__name__)

# Load the pre-trained face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared state for controlling the video feed
video_capture = None
emotion_detection_active = threading.Event()

def generate_frames():
    global video_capture
    video_capture = cv2.VideoCapture(0)  # Using camera index 0

    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if emotion_detection_active.is_set():
            try:
                results = DeepFace.analyze(frame, actions=['emotion'])
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                    emotion = result.get('dominant_emotion', 'No result')
                else:
                    emotion = 'No result'
            except Exception as e:
                print(f"DeepFace analysis error: {e}")
                emotion = "Error"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, emotion, (x, y - 10), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_capture.release()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start')
def start():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_emotion_detection', methods=['POST'])
def toggle_emotion_detection():
    if emotion_detection_active.is_set():
        emotion_detection_active.clear()
    else:
        emotion_detection_active.set()
    return ('', 204)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    # Clear the emotion detection flag
    emotion_detection_active.clear()
    # Release the video capture if it's initialized
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    # Redirect to the home page
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
