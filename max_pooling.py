import cv2
import numpy as np

im   = cv2.imread('focus1.jpg')


cv2.imshow("original", im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

height, width = im.shape[:2]

window_h = 3
window_w = 3

pooling_img   = np.zeros((height/window_h, width/window_w), np.uint8)


for h in range(0,height-window_h-1,window_h):
	
	for w in range(0,width-window_w-1,window_w):
		roi = im[ h:h+window_h, w:w+window_w]
		(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi)
		
		pooling_img[h/window_h,w/window_w] = maxVal
	
cv2.imshow("max pooling", pooling_img)
k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
