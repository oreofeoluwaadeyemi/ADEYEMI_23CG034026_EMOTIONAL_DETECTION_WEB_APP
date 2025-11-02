import sqlite3, os
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(APP_DIR, 'root', 'emotion_detection_db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (id INTEGER PRIMARY KEY, timestamp TEXT, label TEXT, probs TEXT)''')
conn.commit()
conn.close()
print('Initialized DB at', DB_PATH)
