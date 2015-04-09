import cv2
import numpy as np

im   = cv2.imread('focus1.jpg')

def focus_thresh(im):
	gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	focus_thresh = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray_image,3)))
	#print focus_thresh
	return focus_thresh

focus_thresh(im)

cv2.imshow("original", im)

blur = cv2.blur(im,(25,25))
im = cv2.addWeighted(im, 2, blur, -1, 0)

height, width = im.shape[:2]
print height, width
window_h = 3
window_w = 3
for h in range(0,height-window_h-1):
	for w in range(0,width-window_w-1):
		roi = im[ h:h+window_h, w:w+window_w]
		blur_cons = focus_thresh(roi)
		if blur_cons < 100:
			im[ h:h+window_h, w:w+window_w] = cv2.blur(im[ h:h+window_h, w:w+window_w],(10,10))
			#im[ h:h+window_h, w:w+window_w] = cv2.blur(im[ h:h+window_h, w:w+window_w],(35,35))
			im[ h:h+window_h, w:w+window_w] = cv2.medianBlur(im[ h:h+window_h, w:w+window_w],15)			
			#im[ h:h+window_h, w:w+window_w] = cv2.blur(im[ h:h+window_h, w:w+window_w],(15,15))
			#im[ h:h+window_h, w:w+window_w] = cv2.medianBlur(im[ h:h+window_h, w:w+window_w],15)
			#im[ h:h+window_h, w:w+window_w] = cv2.medianBlur(im[ h:h+window_h, w:w+window_w],15)
		#cv2.imshow("original", roi)
		#cv2.waitKey(3000)
#windowsize_r = 8
#windowsize_c = 8

#    		window = gray[c:c+windowsize_c,r:r+windowsize_r]

#for i in range(0,100,2):
#	print i

#roi = im[ 300:500, 300:500]
#roi = im[ 0:200, 0:200]
#focus_thresh(roi)
#im = cv2.blur(im,(3,3))
#im = cv2.medianBlur(im, 3)
cv2.imshow("blur unfocused", im)

k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
