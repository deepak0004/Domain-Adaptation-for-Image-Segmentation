from __future__ import print_function
import cv2
import os

vidcap = cv2.VideoCapture("./Indian_Road/indian_road.mp4")
success,image = vidcap.read()
count = 1
success = True

if not os.path.exists('./Indian_Road/road_frames'):
	os.makedirs('./Indian_Road/road_frames')

while success:
	print("Extracting Frame No. {}".format(count))
	success, image = vidcap.read()
	cv2.imwrite("./Indian_Road/road_frames/img_%04d.png" % count, image)
	count += 1
