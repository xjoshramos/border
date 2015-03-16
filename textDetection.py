import cv2
import numpy as np
import sys
import random

def filter_preprocess(im):
	pre_processed_img = im
	#bilateral_img = cv2.bilateralFilter(im, 11, 25, 25)
        #pre_processed_img = cv2.bilateralFilter(im, 11, 150, 150)
	pre_processed_img = cv2.bilateralFilter(im, 35, 100, 100)
	#blur = cv2.blur(pre_processed_img,(5,5))
	#pre_processed_img = cv2.addWeighted(pre_processed_img, 1.5, blur, -0.5, 0)

	#pre_processed_img = cv2.medianBlur(pre_processed_img, 3)	
	return pre_processed_img
def mser_region_mask(im):
	
	#mask_r = np.zeros((h, w), np.uint8)
	#mask_g = np.zeros((h, w), np.uint8)
	#mask_b = np.zeros((h, w), np.uint8)

	mask_regions = np.zeros((h, w), np.uint8)
	mser = cv2.MSER(8, 50, 500, 0.25, 0.01, 100, 1.01, 0.03, 5)
	
	#r,g,b = cv2.split(im)
	#regions_r = mser.detect(r, None)
	#regions_g = mser.detect(g, None)
	#regions_b = mser.detect(b, None)	

	#for p_r in regions_r:
        #        for pts in p_r:
        #                x, y = pts
        #                mask_r[y,x] = 255
	
	#for p_g in regions_g:
        #        for pts in p_g:
        #                x, y = pts
        #                mask_g[y,x] = 255

	#for p_b in regions_b:
        #        for pts in p_b:
        #                x, y = pts
        #                mask_b[y,x] = 255
	#mask_regions = cv2.bitwise_and(mask_g, mask_r)
	#mask_regions = cv2.bitwise_and(mask_b, mask_regions)
	
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
	
	img = bw_img.copy()	
	filtered_img = np.zeros((h, w), np.uint8)
	contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
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
		if( area > 50 and area < 500 and eccentricity < .983 and solidity > .4):
			cv2.drawContours(filtered_img,[cnt],-1,255,-1)

	return filtered_img

def stroke_to_width_transform(bw_img):
	width_list = []

	return width_list

#im   = cv2.imread('tesseract1.jpg')
im   = cv2.imread('text1.png')
h, w = im.shape[:2]
cv2.imshow('original',im)

im = filter_preprocess(im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 300)
mser_region_img = mser_region_mask(im)
edge_mser_inter = cv2.bitwise_and(edges, mser_region_img)

#TBD gradient growing and mser enahancement
#gradient_grown = grow_edges( gray, edge_mser_inter)

# replace with enhanced image
edge_enhanced_mser = mser_region_img

#filter connected components with hueristics
hFilter_img = filter_hiserstic(edge_enhanced_mser)
hFilter_img = cv2.bitwise_and(hFilter_img, mser_region_img)


cv2.imshow('mser region', edge_mser_inter)
cv2.imshow('filtered',hFilter_img)
cv2.imshow('regions', mser_region_img)
cv2.imshow('preprocessed', im)
cv2.imshow('final candidtates', hFilter_img)

k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
