import argparse
import re
import os
import json

import cv2
import pafy
from tqdm import tqdm

import facedetect


URL_PATTERN = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def detect_faces(face_detector, video, data_dir='data', faces_dir=None, save_every=0,
                 save_min_width=200, save_min_height=200, max_faces=200):
    best = video.getbest()
    # print(f"[INFO] starting video stream... '{video.title}'")
    filename = best.download(filepath="/tmp/", quiet=True)
    # cap = cv2.VideoCapture(best.url)
    cap = cv2.VideoCapture(filename)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    video_length = int(cv2.VideoCapture.get(cap, property_id))
    # time.sleep(1.0)

    if faces_dir is None:
        video_faces_dir = os.path.join(data_dir, video.title)
    else:
        video_faces_dir = os.path.join(data_dir, faces_dir)

    if save_every > 0:
        os.makedirs(video_faces_dir, exist_ok=True)

    # frame_count = 0
    save_count = 0
    for frame_index in tqdm(range(video_length)):
        ret, frame = cap.read()

        if frame is None:
            # print("[INFO] ending video stream...")
            break

        # frame_count += 1
        faces, frame = face_detector.detect(frame)

        if args['save_every'] > 0 and frame_index % save_every == 0:
            for ii, face in enumerate(faces):

                face_shape = face['image'].shape
                if face_shape[0] < save_min_height or face_shape[1] < save_min_width:
                    continue

                cv2.imwrite(os.path.join(video_faces_dir, f'{video.videoid}_frame_{frame_index}__face_{ii}.jpg'),
                            face['image'])
                save_count += 1

        if save_count >= max_faces:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()


def get_video(video_url_or_id):
    if not re.match(URL_PATTERN, video_url_or_id):
        v_url = 'https://www.youtube.com/watch?v=' + video_url_or_id
    return pafy.new(v_url)


def get_playlist(playlist_url_or_id):
    if not re.match(URL_PATTERN, playlist_url_or_id):
        p_url = 'https://www.youtube.com/playlist?list=' + playlist_url_or_id
    return pafy.get_playlist(p_url)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-v", "--video", help="A youtube ID or URL")
    group.add_argument("-p", "--playlist", help="A youtube playlist ID or URL")
    group.add_argument("-j", "--json", help="Path for a JSON file with URLs")

    ap.add_argument("--playlist-start", type=int, default=0,
                    help="start index for videos in a playlist")
    ap.add_argument("--playlist-end", type=int,
                    help="end index for videos in a playlist")
    ap.add_argument("-t", "--prototxt", default='models/deploy.prototxt',
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default='models/res10_300x300_ssd_iter_140000.caffemodel',
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--save-every", type=int, default=0,
                    help="save detected faces to files every N frames. If zero, do not save face images.")
    ap.add_argument("-d", "--data-dir", type=str, default='data',
                    help="directory to save faces")
    args = vars(ap.parse_args())

    detector = facedetect.FaceDetector(args["prototxt"], args["model"])

    if args.get('json'):
        with open(args['json'], 'r') as f:
            url_groups = json.load(f)

            pbar_groups = tqdm(url_groups.items())
            for group, urls in pbar_groups:
                pbar_groups.set_description(f'Processing group "{group}"')

                pbar_urls = tqdm(urls)
                for url_or_id in pbar_urls:
                    video = get_video(url_or_id)
                    pbar_urls.set_description(f'Detecting faces in "{video.title}"')
                    detect_faces(detector, video, data_dir=args['data_dir'],
                                 faces_dir=group, save_every=args['save_every'])

    elif args.get('playlist'):
        playlist = get_playlist(args['playlist'])
        pbar = tqdm(playlist['items'][args['playlist_start']:args.get('playlist_end')])

        for video in pbar:
            video_title = video["pafy"].title
            pbar.set_description(f'Detecting faces in "{video_title}"')
            detect_faces(detector, video['pafy'], data_dir=args['data_dir'], save_every=args['save_every'])
    else:
        video = get_video(args['video'])
        detect_faces(detector, video, data_dir=args['data_dir'], save_every=args['save_every'])

    cv2.destroyAllWindows()
