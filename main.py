import cv2
import time
import base64
import json
import os
from moviepy.editor import VideoFileClip
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
from PIL import Image
from openai import OpenAI
#import dlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

import pathlib


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

CORS(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

client = OpenAI(api_key ='settings')
MODEL="gpt-4o"
#detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
conf_thres = 0.6

model = load_model('vgg16_10epoch_face_cnn_model_v2.h5')


cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)

face_label_filename = 'vgg16_10epoch_face_label_v2.pickle'
with open(face_label_filename, "rb") as f: class_dictionary = pickle.load(f)
class_list = [value for _, value in class_dictionary.items()]


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
obj_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_v5_3.pt')


def extract_frames(video_path, seconds_per_frame=1):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

def detect_purchase(model, frames):
  response = client.chat.completions.create(
      model=model,
      messages=[
      {"role":"system", "content":"You are a machine that only returns and replies with valid, iterable RFC8259 compliant JSON in your responses" },
      {"role": "system", "content": "You are classifying a video as whether the purchase of item(s) occured. Return a json without any '```json' with a description field summarizing the video and a tansaction field indicating a True or False of if a purchase occured."},
      {"role": "user", "content": [
          "These are the frames from the video.",
          *map(lambda x: {"type": "image_url",
                          "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, frames)
          ],
      }
      ],
      temperature=0,
  )
  json_response = response.choices[0].message.content
  json_response = json.loads(json_response.strip())
  return json_response.get('transaction')


def decode_base64_frame(base64_frame):
    decoded_data = base64.b64decode(base64_frame)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return image

# def extractFace(image, x1, x2, y1, y2):
#     image_array = np.asarray(image, "uint8")
#     y_min = min(y1, y2)
#     y_max = max(y1, y2)
#     x_min = min(x1, x2)
#     x_max = max(x1, x2)
#     face = image_array[y_min:y_max, x_min:x_max]
#     try:
#         face = cv2.resize(face, (224, 224) )
#         face_array = np.asarray(face,  "uint8")
#         return face_array
#     except:
#         return None
    

# def detectFace(image):
#     image_array = np.asarray(image, "uint8")
#     faces_detected = detector(image_array)
#     if len(faces_detected) == 0:
#         return []
#     faces_extracted = []

#     for face in faces_detected:

#         conf = face.confidence
#         if conf < conf_thres:
#             continue

#         x1 = face.rect.left()
#         y1 = face.rect.bottom()
#         x2 = face.rect.right()
#         y2 = face.rect.top()


#         face_array = extractFace(image, x1, x2, y1, y2)
#         if face_array is not None:
#             faces_extracted.append(face_array)

#     return faces_extracted

# def preprocess_image(image):
#     image = cv2.resize(image, (224, 224))
#     image = image.astype('float32')
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image

# def preprocess_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#     return cv2.equalizeHist(gray) 


# def detect_faces_mmod(frame):
#     rgb_frame = preprocess_frame(frame)
#     detections = detector(rgb_frame, 2)
#     face_locations_mmod = [(d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left()) for d in detections]
#     return face_locations_mmod

def detect_object(frames):
  label = ""
  object_detection_count = 0
  for frame in frames:
    frame = decode_base64_frame(frame)
    results = obj_detection_model(frame)
    for det in results.xyxy[0]:
      x1, y1, x2, y2, conf, cls = det
      if label != "":
        prev_score = float(label.split(" ")[1])
        if conf>=0.4 and conf > prev_score:
          label = f'{obj_detection_model.names[int(cls)]} {conf:.2f}'
      object_detection_count += 1
      label = f'{obj_detection_model.names[int(cls)]} {conf:.2f}'
    if object_detection_count>=2:
      break
    return label.split(" ")[0]
# def predict_faces(frame, model, class_names):
#     face_locations = detect_faces_mmod(frame)
#     face_names = []
#     for (top, right, bottom, left) in face_locations:
#         face_image = frame[top:bottom, left:right]
#         face_image = preprocess_image(face_image)

#         predictions = model.predict(face_image)
#         best_class = np.argmax(predictions)
#         name = class_names[best_class]
#         face_names.append(name)
#     return face_names

# def identify_people(frames):
#   buyer, seller = None, None
#   face_detection_count = 0
#   face_names = []
#   for frame in frames:
#     frame = decode_base64_frame(frame)
#     face_names =  predict_faces(frame, model, class_list)
#     if (len(face_names) >= 2):
#       face_detection_count += 1
#     if face_detection_count>=2:
#       break
#   for name in face_names:
#     if "buyer" in name.lower():
#       buyer = name.split("-")[0]
#     elif "seller" in name.lower():
#       seller = name.split("-")[0]
#   return buyer, seller

def update_database(buyer, seller, product):
   db = firestore.client()
   doc_ref = db.collection("transactions").document()
   doc_ref.set({"cashier": seller, "buyer": buyer, "product": product  })

def process_video(video_path):
   frames = extract_frames(video_path)
   #is_transaction = detect_purchase(MODEL, frames)
   is_transaction = True
   if is_transaction == True:
      product = detect_object(frames)
    # buyer, seller = identify_people(frames)
      # update_database("Test", "Test", product)
      return    


@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file path"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        thread = Thread(target=process_video, args=(file_path,))
        thread.start()
        return jsonify({"message": "File uploaded successfully. Processing started."}), 202


if __name__ == '__main__':
    app.run(debug=True)
