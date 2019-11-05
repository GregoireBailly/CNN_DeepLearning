import imutils
import argparse
import cv2
import time



def pyramid(image, scale=1.2, minSize=(36,36)):
    yield image

    while True:
        #compute the new dimensions and resize
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)

        #if image is small enough stop
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


def sliding_window(image, stepSize, windowSize):
    #sliding a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range (0, image.shape[1], stepSize):
            #yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


image = cv2.imread("C:\\Users\\Greg\\Documents\\0.Work Hard\\0.INSA\\5A\\IA\\image1.jpg")
(winW, winH) = (36, 36)

#loop over the pyramid
for (i, resized) in enumerate(pyramid(image)):
    # loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=15, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
 
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)

cv2.destroyAllWindows()