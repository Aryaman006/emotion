from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
from deepface import DeepFace
import threading
import requests
import base64

# Spotify API credentials
SPOTIFY_CLIENT_ID = 'your-client-id'  # 🔁 Replace with your actual Client ID
SPOTIFY_CLIENT_SECRET = 'your-client-secret'  # 🔁 Replace with your actual Client Secret

app = Flask(__name__)

# Load the pre-trained face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared state
video_capture = None
emotion_detection_active = threading.Event()

# Current detected emotion and suggested song
current_song = {
    "emotion": "neutral",
    "song": "Let Her Go",
    "artist": "Passenger",
    "url": "https://open.spotify.com/track/3ZFTkvIE7kyPt6Nu3PEa7V"
}


def get_spotify_token():
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials"
    }

    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    token = response.json().get("access_token")
    return token


def get_song_from_spotify(emotion):
    token = get_spotify_token()
    headers = {
        "Authorization": f"Bearer {token}"
    }

    query = f"{emotion} mood song"
    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }

    response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
    items = response.json().get("tracks", {}).get("items", [])
    if items:
        track = items[0]
        return {
            "song": track["name"],
            "artist": track["artists"][0]["name"],
            "url": track["external_urls"]["spotify"]
        }
    else:
        return {
            "song": "No suggestion",
            "artist": "N/A",
            "url": "#"
        }


def generate_frames():
    global video_capture, current_song
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if emotion_detection_active.is_set():
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = results[0]['dominant_emotion'] if isinstance(results, list) else results.get('dominant_emotion', 'No result')
            except Exception as e:
                print(f"DeepFace error: {e}")
                emotion = "Error"

            if emotion and emotion != current_song.get('emotion'):
                current_song['emotion'] = emotion
                current_song.update(get_song_from_spotify(emotion))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
    emotion_detection_active.clear()
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    return ('', 204)


@app.route('/current_song')
def get_current_song():
    return jsonify(current_song)


if __name__ == '__main__':
    app.run(debug=True)
