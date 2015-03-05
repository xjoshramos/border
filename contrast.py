import cv2
import numpy as np

im   = cv2.imread('bordered3.jpg')

depth = im.shape

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

if len(depth) == 3:
	r,g,b = cv2.split(im)

	b_contrast = clahe.apply(b)
	g_contrast = clahe.apply(g)
	r_contrast = clahe.apply(r)

	image_contrast = cv2.merge([r_contrast,g_contrast,b_contrast])

if len(depth) == 2:
	image_contrast = clahe.apply(im)

cv2.imshow("original" , im)
cv2.imshow("contrast" , image_contrast)

k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
