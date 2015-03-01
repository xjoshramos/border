import cv2
import numpy as np

im = cv2.imread('bordered2.jpeg')
#im = cv2.medianBlur(im, 3)
height, width = im.shape[:2]

print im.shape
print 'height %d width %d' % (height, width)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
grayThresh,otsu = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

edges = cv2.Canny(gray, grayThresh*0.5, grayThresh)


while(True):

    img = im.copy()
    stackX = []
    stackY = []

    for i in range(0, height):
    	for j in range(0, width):
		pixVal = edges[i,j]
		if pixVal == 255:
			stackY.append(i)
			stackX.append(j)

    x1 = min(stackX)+1
    y1 = min(stackY)+1
    x2 = max(stackX)
    y2 = max(stackY)

    print  'x1 = %d y1 = %d x2 = %d y2 = %d ' % (x1, y1, x2, y2)
    roi = im[ y1:y2, x1:x2] 
   
    #cv2.imshow('houghlines',edges)
    #cv2.imshow('hue',hueOtsu)
    #cv2.imshow('otsu',otsu)
    cv2.imshow("lines" , img)
    cv2.imshow("roi", roi)	    


    k = cv2.waitKey(0)
    if k == 27:
        break

cv2.destroyAllWindows()
