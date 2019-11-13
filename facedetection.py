# from https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
# from __future__ import print_function
import cv2 as cv
import argparse


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect fist
    fists = fist_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in fists:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        print("Fist")
    # -- Detect palm
    palms = palm_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in palms:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        print("Palm")
    cv.imshow('Capture - Face detection', frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--fist_cascade', help='Path to gest cascade.', default='data/haarcascades/fist.xml')
parser.add_argument('--palm_cascade', help='Path to gest cascade.', default='data/haarcascades/palm_v4.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
fist_cascade_name = args.fist_cascade
palm_cascade_name = args.palm_cascade
fist_cascade = cv.CascadeClassifier()
palm_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
if not fist_cascade.load(cv.samples.findFile(fist_cascade_name)):
    print('--(!)Error loading fist cascade')
    exit(0)
if not palm_cascade.load(cv.samples.findFile(palm_cascade_name)):
    print('--(!)Error loading palm cascade')
    exit(0)
camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
