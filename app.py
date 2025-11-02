from flask import Flask, render_template, request, jsonify
import os, io, base64, sqlite3, datetime, json
from PIL import Image
import numpy as np
import cv2
from model import EmotionModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'root', 'emotion_detection_db')
os.makedirs(os.path.join(APP_DIR, 'root'), exist_ok=True)

app = Flask(__name__)
model = EmotionModel()  # loads emotion_mlp_model1.pkl

def init_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

@app.route('/')
def index():
    return render_template('index.html')

def save_prediction(label, probs):
    conn = init_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, timestamp TEXT, label TEXT, probs TEXT)''')
    ts = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO predictions (timestamp, label, probs) VALUES (?, ?, ?)",
              (ts, label, json.dumps(probs)))
    conn.commit()
    conn.close()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({'error':'no image provided'}), 400
    header, encoded = img_b64.split(',', 1) if ',' in img_b64 else ('', img_b64)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)
    label, probs = model.predict_from_array(img_np)
    try:
        save_prediction(label, probs)
    except Exception:
        pass
    return jsonify({'label': label, 'probs': probs})

if __name__ == '__main__':
    print('Starting server on http://127.0.0.1:5000')
    app.run(debug=True)
