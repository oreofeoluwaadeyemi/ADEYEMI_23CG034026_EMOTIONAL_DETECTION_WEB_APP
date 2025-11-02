import os, pickle, cv2, numpy as np
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'emotion_mlp_model1.pkl')
LABELS = ['Happy','Sad','Neutral']
class EmotionModel:
    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self._load_model()
        if hasattr(cv2.data, 'haarcascades'):
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:
            self.cascade_path = 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
    def _load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f'[Warning] Could not load model at {self.model_path}:', e)
            self.model = None
    def _preprocess(self, face_img):
        if face_img is None:
            return None
        if face_img.ndim == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        else:
            face_gray = face_img
        face_resized = cv2.resize(face_gray, (48,48), interpolation=cv2.INTER_AREA)
        face_norm = face_resized.astype('float32') / 255.0
        return face_norm.flatten().reshape(1, -1)
    def predict_from_array(self, img_array):
        img = img_array.copy()
        if img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return ('NoFaceDetected', {})
        x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        face_region = gray[y:y+h, x:x+w]
        X = self._preprocess(face_region)
        if X is None:
            return ('Error', {})
        if self.model is None:
            return ('ModelNotLoaded', {})
        probs = self.model.predict_proba(X)[0].tolist()
        pred_idx = int(self.model.predict(X)[0])
        label = LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx)
        prob_map = {LABELS[i]: float(probs[i]) for i in range(min(len(probs), len(LABELS)))}
        return (label, prob_map)
