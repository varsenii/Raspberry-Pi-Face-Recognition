import face_recognition

import pickle
import cv2
import os
import argparse

from aiutils.filesystem import get_image_paths

from config import DATASET_PATH, ENCODINGS_PATH

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--detection_method', type=str, default='cnn',
                help="face detection mthod to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())
print(f'[INFO] detection method: {args["detection_method"]}')

print('[INFO] quantifying faces...')
image_paths = get_image_paths(DATASET_PATH)
# initialize the list of known names and encodings
known_encodings = []
known_names = []

# loop over the input image paths
for i, image_path in enumerate(image_paths):
    print(f'[INFO] processing image {i+1}/{len(image_paths)}')

    # load the image and convert it from BGR (OpenCV ordering) to RGB (dlib ordering)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imshow('fs', rgb_image)

    # detect the (x, y)-coordinates of the bounding boxes
    boxes = face_recognition.face_locations(rgb_image, model=args['detection_method'])
    
    # print(boxes)
    # for top, right, bottom, left in boxes:
    #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    # cv2.imshow('image', cv2.resize(image, (int(image.shape[0]/2), int(image.shape[1]/2))))
    # cv2.waitKey(0)
    # cv2.destroyWindow('image')
    # continue

    # compute the facial embeddings
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # add the the name and face encodings to the list
    name = image_path.split(os.path.sep)[-2]
    for encooding in encodings:
        known_encodings.append(encooding)
        known_names.append(name)

# dump the facial encodings to the disk
print('[INFO] serializing encodings...')
data = {'encodings':known_encodings, 'names':known_names}
with open(ENCODINGS_PATH, 'wb') as f:
    f.write(pickle.dumps(data))