from flask import Flask, request, jsonify
import os
from threading import Thread
import firebase_admin
from firebase_admin import firestore

# Application Default credentials are automatically created.
# app = firebase_admin.initialize_app()
# db = firestore.client()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


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
    
        # thread = Thread(target=process_video, args=(file_path,))
        # thread.start()
        return jsonify({"message": "File uploaded successfully. Processing started."}), 202
    

if __name__ == '__main__':
    app.run(debug=True)