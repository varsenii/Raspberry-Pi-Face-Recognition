import time
import cv2

from config import CASCADE_PATH, BOX_COLOR
from aiutils.images.preprocessing import resize

print('[INFO] loading the OpenCV\'s Haar Cascades')
detector = cv2.CascadeClassifier(CASCADE_PATH)

print('[INFO] starting video stream...')
vid = cv2.VideoCapture(0)
start_time = time.time()
n_frames = 0
while True:
    ret, frame = vid.read()
    if not ret:
        break
    n_frames += 1

    # grab the frame from the threaded video stream and resize it to 500px (to speedup processing)
    frame = resize(frame, width=500)

    # convert the input image frame from BGR to grayscale for face detection
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray_image, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # display face bounding boxes along with the names
    for (top, righ, bottom, left) in boxes:
        cv2.rectangle(frame, (left, top), (righ, bottom), BOX_COLOR, 2)
        y = top - 15 if top - 15 > 15 else top + 15
    
    cv2.imshow('webcam', frame)
    # Wait for 1 millisecond and check for user input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break