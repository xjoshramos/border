import cv2
import numpy as np

im   = cv2.imread('bordered3.jpg')

depth = im.shape


#if len(depth) == 3:
blur = cv2.blur(im,(5,5))
img_sharp = cv2.addWeighted(im, 1.5, blur, -0.5, 0)
#if len(depth) == 2:
	

cv2.imshow("original" , im)
cv2.imshow("contrast" , img_sharp)

k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
