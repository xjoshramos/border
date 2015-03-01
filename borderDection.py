import cv2
import numpy as np

im   = cv2.imread('bordered7.jpeg')
h, w = im.shape[:2]

mask1   = np.zeros((h+2, w+2), np.uint8)
seedPt1 = (0,0)

mask2   = np.zeros((h+2, w+2), np.uint8)
seedPt2 = (w-1,h-1);

mask    = np.zeros((h+2, w+2), np.uint8)
drawing = np.zeros((h, w), np.uint8)

diff = (2,2,2)
flag = 8|255<<8

cv2.floodFill(im, mask1, seedPt1, (255,0,0),diff,diff,flag|cv2.FLOODFILL_MASK_ONLY) 
cv2.floodFill(im, mask2, seedPt2, (0,0,255),diff,diff,flag|cv2.FLOODFILL_MASK_ONLY)

mask = cv2.bitwise_or(mask1,mask2)

border = False

contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	#cv2.drawContours(im,[cnt],-1,(0,0,255),3)
	approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
	print len(approx)
	if len(approx) == 4:
		cv2.drawContours(drawing,[cnt],-1,255,-1)



if drawing[h-1,1] and drawing[1,w-1] == 255 and  drawing[1,1] == 255 and drawing[h-1,w-1] == 255:
	border = True

print drawing[1,1]
print drawing[h-1,w-1]
print drawing[h-1,1]
print drawing[1,w-1]
print border

cv2.imshow("lines" , drawing)

k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
