import pafy
import cv2
import argparse
import time
import facedetect


ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True,
                help="The youtube URL to process")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

face_detector = facedetect.FaceDetector(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
url = args['url']
video = pafy.new(url)
play = video.getbest()
cap = cv2.VideoCapture(play.url)
time.sleep(2.0)

while True:
    ret, frame = cap.read()

    if frame is None:
        print("[INFO] ending video stream...")
        break

    faces, frame = face_detector.detect(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
