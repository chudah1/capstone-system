import cv2
import time
from collections import deque
import os
import requests


def capture_video():
    cap = cv2.VideoCapture(0)
    frame_buffer = deque()
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)
            if time.time() - start_time >= 10:
                video_path = "transaction_video.mp4"
                combine_frames_to_video(list(frame_buffer), video_path)
                status_code, response_text = upload_video_to_api(video_path, "http://127.0.0.1:5000/process_video")
                print(f"Uploaded with status code {status_code}: {response_text}")
                frame_buffer.clear()
                start_time = time.time()
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def combine_frames_to_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

def upload_video_to_api(video_path, api_endpoint):
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_endpoint, files=files)
    return response.status_code, response.text

if __name__ == "__main__":
    capture_video()