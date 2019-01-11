import pafy
import cv2
import argparse
import time
import facedetect
import re
import os
from tqdm import tqdm

URL_PATTERN = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def detect_faces(face_detector, video, data_dir='data', save_every=0):
    play = video.getbest()
    # print(f"[INFO] starting video stream... '{video.title}'")
    cap = cv2.VideoCapture(play.url)
    time.sleep(1.0)

    frame_count = 0
    video_faces_dir = os.path.join(data_dir, video.title)

    if save_every > 0:
        os.makedirs(video_faces_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if frame is None:
            # print("[INFO] ending video stream...")
            break

        frame_count += 1
        faces, frame = face_detector.detect(frame)

        if args['save_every'] > 0 and frame_count % save_every == 0:
            for ii, face in enumerate(faces):
                cv2.imwrite(os.path.join(video_faces_dir, f'frame_{frame_count}__face_{ii}.jpg'), face['image'])

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-v", "--video", help="A youtube ID or URL")
    group.add_argument("-p", "--playlist", help="A youtube playlist ID or URL")
    ap.add_argument("--playlist-start", type=int, default=0,
                    help="start index for videos in a playlist")
    ap.add_argument("--playlist-end", type=int,
                    help="end index for videos in a playlist")
    ap.add_argument("-t", "--prototxt", required=True,
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

    detector = facedetect.FaceDetector(args["prototxt"], args["model"])

    p_url = args.get('playlist')
    if p_url:
        if not re.match(URL_PATTERN, args['playlist']):
            p_url = 'https://www.youtube.com/playlist?list=' + args['playlist']
            playlist = pafy.get_playlist(p_url)

            pbar = tqdm(playlist['items'][args['playlist_start']:args.get('playlist_end')])

            for video in pbar:
                video_title = video["pafy"].title
                pbar.set_description(f'Processing video "{video_title}"')
                detect_faces(detector, video['pafy'], data_dir=args['data_dir'], save_every=args['save_every'])
    else:
        v_url = args['video']
        if not re.match(URL_PATTERN, v_url):
            v_url = 'https://www.youtube.com/watch?v=' + v_url
        video = pafy.new(v_url)
        detect_faces(detector, video, data_dir=args['data_dir'], save_every=args['save_every'])

    cv2.destroyAllWindows()
