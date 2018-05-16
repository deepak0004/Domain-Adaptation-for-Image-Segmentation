import cv2
import os

background_files = sorted(os.listdir("./Indian_Road/road_frames/."))
overlay_files = sorted(os.listdir("./Indian_Road/segmented_output/."))

for file in background_files:
	print("Processed: " + file)
	background = cv2.imread("./Indian_Road/road_frames/" + file)
	background = cv2.resize(background, (2048, 1024), interpolation = cv2.INTER_CUBIC)
	overlay = cv2.imread("./Indian_Road/segmented_output/" + file)

	added_image = cv2.addWeighted(background,1,overlay,0.99,0)
	cv2.imwrite('./Indian_Road/overlay_output/' + file, added_image)