import cv2
import numpy as np


class FaceDetector:
    def __init__(self, prototxt, model):
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, frame, min_confidence=0.5, draw_boxes=True, draw_text=True):
        """
        Detect faces present in a given image using a pre-trained deep learning model.

        :param frame: the image to be processed
        :param min_confidence: minimum confidence to filter detections
        :param draw_boxes: if True, draw bounding boxes for detected faces
        :param draw_text: if True, draw text informing detection confidence and face count.
        :return: tuple (faces, frame) with coordinates and confidence of each face detection.
        """

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        count = 0
        faces = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            face = {}

            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # print(confidence * 100)
            face['confidence'] = confidence * 100

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < min_confidence:
                continue

            count += 1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face['box'] = box.astype("int")
            faces.append(face)

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100) + ", Count " + str(count)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            if draw_boxes:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            if draw_text:
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return faces, frame
