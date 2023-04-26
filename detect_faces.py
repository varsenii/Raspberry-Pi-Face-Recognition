import time
import cv2

from config import CASCADE_PATH, BOX_COLOR
from aiutils.images.preprocessing import resize_to_max_width
from aiutils.video import FPS, WebcamVideoStream

print('[INFO] loading the OpenCV\'s Haar Cascades')
detector = cv2.CascadeClassifier(CASCADE_PATH)

print('[INFO] starting video stream...')
vs = WebcamVideoStream(0).start()
fps = FPS().start()
while True:
    # grab the frame from the threaded video stream and resize it to 500px (to speedup processing)
    frame = vs.read()
    frame = resize_to_max_width(frame, max_width=500)

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

    fps.update()

fps.stop()
print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
print('[INFO] aprox. FPS: {:.2f}'.format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()