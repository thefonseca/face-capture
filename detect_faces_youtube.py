import pafy
import cv2
import argparse
import time
import facedetect
import re
import os

URL_PATTERN = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="The youtube ID or URL")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save-every", type=int, default=0,
                help="save detected faces to files every N frames. If zero, do not save face images.")
ap.add_argument("-d", "--data-dir", type=str, default='data',
                help="directory to save faces")
args = vars(ap.parse_args())

face_detector = facedetect.FaceDetector(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
video = args['video']

if not re.match(URL_PATTERN, video):
    url = 'https://www.youtube.com/watch?v=' + video

video = pafy.new(video)
play = video.getbest()
cap = cv2.VideoCapture(play.url)
time.sleep(2.0)

frame_count = 0
video_faces_dir = os.path.join(args['data_dir'], video.title)

if args['save_every'] > 0:
    os.makedirs(video_faces_dir, exist_ok=True)

while True:
    ret, frame = cap.read()

    if frame is None:
        print("[INFO] ending video stream...")
        break

    frame_count += 1
    faces, frame = face_detector.detect(frame)

    if args['save_every'] > 0 and frame_count % args['save_every'] == 0:
        for ii, face in enumerate(faces):
            cv2.imwrite(os.path.join(video_faces_dir, f'frame_{frame_count}__face_{ii}.jpg'), face['image'])

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
