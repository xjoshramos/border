import cv2
import numpy as np
import sys
import random

r = random.randint(0, 255)
g = random.randint(0, 255)
b = random.randint(0, 255)


def mser_region_mask(im):
	mask_regions = np.zeros((h, w), np.uint8)
	mser = cv2.MSER(8, 50, 500, 0.25, 0.01, 100, 1.01, 0.03, 5)
	
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	regions = mser.detect(gray, None)

	for p in regions:
        	for pts in p:
                	x, y = pts
                	mask_regions[y,x] = 255

	return mask_regions

def grow_edges(gray, edge_mser_inter):
	gradient_grown = gray	
	grad_x = cv2.Sobel(gray,cv2.CV_32F,1,0)
        grad_y = cv2.Sobel(gray,cv2.CV_32F,0,1)
	
	grad_mag = grad_x
	grad_dir = grad_x
	#grad_mag = np.zeros((h, w), cv2.CV_32F)
	#grad_dir = np.zeros((h, w), cv2.CV_32F)

	cv2.cartToPolar(grad_x, grad_y, grad_mag, grad_dir, True)
	
	#for y in range(0,h):
		#grad_dir
		#grad_ptr = grad_dir
	
	return gradient_grown

def filter_hiserstic(bw_img):
	
	filtered_img = np.zeros((h, w), np.uint8)
	contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	solidity = 0
	eccentricity = 1
	#.995
	for cnt in contours:
		area = cv2.contourArea(cnt)
		hull = cv2.convexHull(cnt)
		hull_area = cv2.contourArea(hull)
		if hull_area > 0:
			solidity = float(area)/hull_area
		if len(cnt) > 5:
			ellipse = cv2.fitEllipse(cnt)
			center, axes, orientation = ellipse
			majoraxis_len =	max(axes)
			minoraxis_len = min(axes)
			if majoraxis_len > 0:
				eccentricity = np.sqrt(1-(minoraxis_len/majoraxis_len)**2)
		if( area > 50 and area < 2000 and eccentricity < .983 and solidity > .4):
			cv2.drawContours(filtered_img,[cnt],-1,255,1)

	return filtered_img

im   = cv2.imread('tesseract1.jpg')
#im   = cv2.imread('text1.png')
h, w = im.shape[:2]


gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 300)
mser_region_img = mser_region_mask(im)
edge_mser_inter = cv2.bitwise_and(edges, mser_region_img)

#TBD gradient growing and mser enahancement
#gradient_grown = grow_edges( gray, edge_mser_inter)

# replace with enhanced image
edge_enhanced_mser = mser_region_img

#filter connected components
edge_enhance_mser = filter_hiserstic(edge_enhanced_mser)

cv2.imshow('mser region', edge_mser_inter)
cv2.imshow('filtered',edge_enhance_mser)



k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
