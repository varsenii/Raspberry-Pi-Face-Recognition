import face_recognition
import argparse
import pickle
import time
import cv2

from config import CASCADE_PATH, ENCODINGS_PATH, BOX_COLOR
from aiutils.images.preprocessing import resize_to_max_width
from aiutils.video import WebcamVideoStream, FPS

print('[INFO] loading the embeddings of the known faces along with OpenCV\'s Haar Cascades')
data = pickle.loads(open(ENCODINGS_PATH, 'rb').read())
detector = cv2.CascadeClassifier(CASCADE_PATH)

print('[INFO] starting video stream...')
vs = WebcamVideoStream(0).start()
fps = FPS().start()
while True:
    frame = vs.read()

    # grab the frame from the threaded video stream and resize it to 500px (to speedup processing)
    frame = resize_to_max_width(frame, max_width=500)

    # convert the input image frame from BGR to grayscale for face detection
    # and to RGB for face recognition
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray_image, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute facial embeddings for each face..
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    names = []
    for encoding in encodings:
        # searh for match between each face in the input image and the known encodings
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = 'unknown'

        if True in matches:
            # count the total number of matches for each known person
            match_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in match_idxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face, that is the one whith the largest number of votes
            name = max(counts, key=counts.get)
        names.append(name)

    # display face bounding boxes along with the names
    for ((top, righ, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (righ, bottom), BOX_COLOR, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BOX_COLOR, 2)

    cv2.imshow('webcam', frame)
    # Wait for 1 millisecond and check for user input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    fps.update()
    print(f'FPS: {fps.fps():.2f}')

vs.stop()
cv2.destroyAllWindows()