import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from deepface import DeepFace
import threading
import json
from scipy.spatial.distance import cosine
from datetime import datetime
import csv

frame = None
faces_detected = []
names = []
lock = threading.Lock()

cache_file = "embeddings_cache.json"

_dir = "attendance"
_month = str(datetime.now().strftime("%B"))
_day = str(datetime.today().strftime("%Y-%m-%d"))
attendance_file = ""

with open(cache_file, 'r') as f:
    database = json.load(f)

print(f"Database loaded")

def create_attendance():
    global _dir, _month, attendance_file, names

    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    current_dir = os.path.join(_dir, _month)        
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)

    attendance_file = os.path.join(current_dir, _day)
    attendance_file = attendance_file + ".csv"

    with open(attendance_file, 'a') as csvfile:
        r = csv.writer(csvfile)

        if os.path.getsize(attendance_file) == 0:
            r.writerow(['name', 'date', 'time'])
        else:
            with open(attendance_file, 'r') as csvfile:
                r = csv.reader(csvfile)
                for row in r:
                    if len(row) > 0:
                        names.append(row[0])


def make_attendance(name):
    global attendance_file
    date = str(datetime.today().strftime("%Y-%m-%d"))
    time = str(datetime.now().time().strftime("%H:%M"))

    with open(attendance_file, 'a') as csvfile:
        r = csv.writer(csvfile)

        r.writerow([name, date, time])


def find_best_match(embedding):
    name = ""
    best_distance = float('inf')

    for person in database:
        distance = cosine(embedding, person['embedding'])

        if distance < best_distance:
            best_distance = distance
            name = person['name']

    return name, best_distance


def search_face():
    global frame, faces_detected

    while True:
        with lock:
            if frame is None:
                continue
            frameCopy = frame.copy()

        try:
            resize_scale = 4
            new_resize = resize_scale / (pow(resize_scale, 2))   
            frameS = cv2.resize(frameCopy, (0, 0), None, new_resize, new_resize)
            
            faces = DeepFace.extract_faces(
                img_path=frameS,
                detector_backend='mtcnn',
                enforce_detection=False
            )

            new_faces = []

            for face in faces:
                if face['confidence'] == 0:
                    continue

                embedding = DeepFace.represent(
                    img_path=face['face'],
                    model_name='ArcFace',
                    detector_backend='skip',
                    enforce_detection=False
                )

                current_embedding = embedding[0]['embedding']

                name, distance = find_best_match(current_embedding)

                face_area = face['facial_area']
                new_faces.append({
                    'name': name,
                    'x': face_area['x'] * resize_scale,
                    'y': face_area['y'] * resize_scale,
                    'w': face_area['w'] * resize_scale,
                    'h': face_area['h'] * resize_scale
                })

            faces_detected = new_faces

        except Exception as e:
            print("Erro:", e)
            faces_detected = []


if __name__ == "__main__":
    thread = threading.Thread(target=search_face, daemon=True)
    thread.start()

    create_attendance()

    cap = cv2.VideoCapture(2)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    
    while True:
        success, newFrame = cap.read()

        if not success:
            continue

        with lock:
            frame = newFrame

        for face in faces_detected:
            cv2.rectangle(newFrame, 
                        (face['x'], face['y']), 
                        (face['w'] + face['x'], face['h'] + face['y']), 
                        (255, 0, 255), 2)

            

            cv2.putText(newFrame, str(face['name']), (face['x'] - 50, face['y'] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if face['name'] not in names:
                make_attendance(face['name'])
                names.append(face['name'])
            
        cv2.imshow('Webcam', newFrame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

        
