import cv2
import numpy as np

import sys
import random

import random
import math

def filter_preprocess(im):
	#pre_processed_img = im
	pre_processed_img = cv2.bilateralFilter(im, 35, 100, 100)
	return pre_processed_img
def mser_region_mask(im):
	

	mask_regions = np.zeros((h, w), np.uint8)
	mser = cv2.MSER(8, 25, 500, 0.25, 0.01, 100, 1.01, 0.03, 5)
	
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)	
	regions = mser.detect(gray, None)

	for p in regions:
        	for pts in p:
                	x, y = pts
                	mask_regions[y,x] = 255

	return mask_regions

def gradient_img(gray):

	sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
        sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

	phase = sobel_x
	energy = cv2.magnitude(sobel_x, sobel_y)
	phase  = cv2.phase(sobel_x, sobel_y, phase, False)
	cv2.normalize(energy, energy, 0.0, 1.0, cv2.NORM_MINMAX);
	
	return energy, phase

def aspect_ratio_contour(contour):
	
	x,y,w,h = cv2.boundingRect(cnt)
	aspect_ratio = float(w)/h
	return aspect_ratio

def extent_contour(contour):

	area = cv2.contourArea(contour)
	x,y,w,h = cv2.boundingRect(contour)
	rect_area = w*h
	extent = float(area)/rect_area
	return extent

def solidity_contour(contour):
	area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
		s = float(area)/hull_area
	else:
		s = 0
	return s

def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)

def eccentricity_contour(contour):
	
	if len(contour) > 5:

		ellipse = cv2.fitEllipse(contour)
                center, axes, orientation = ellipse
                majoraxis_len = max(axes)
                minoraxis_len = min(axes)
                if majoraxis_len > 0:
			e = np.sqrt(1-(minoraxis_len/majoraxis_len)**2)
	else: 
		e = 1

	return e
def mean_std_dev(rect_img, area):
	h1, w1 = rect_img.shape[:2]
	count = 0
	mean_sum = 0
	mean = 0
	std_dev = 1
	ratio = 0
	for x in range(0, h1 -1):
		for y in range(0, w1 - 1):
			if rect_img[x,y] > 0:
				count += 1
				mean_sum += rect_img[x,y]
	if count > 0:
		mean = mean_sum/count
		
	accum = 0
        for x in range(0,h1-1):
        	for y in range(0,w1-1):
			if rect_img[x,y] > 0:
				accum += (rect_img[x,y] - mean)*( rect_img[x,y] - mean)
        if count > 0:
		std_dev= np.sqrt(accum/count)
        if mean > 0:
		ratio = std_dev/mean
	
	#data = cv2.meanStdDev(rect_img)
	#mean_cv, std_d_cv = data
	#ratio_i = std_d_cv/mean_cv
	#print mean_cv, std_d_cv, ratio_i
	
	return mean, std_dev, ratio 

def grad_stw(bw_img, contour, bw_mask):
	stw_mag = []
	grad_mag, grad_dir = gradient_img(bw_img)
	k = 0
	contour = np.vstack(contour).squeeze()
	for pt in contour:
		debug = bw_mask.copy()*255
		stroke_width = 0
		x1, y1 =  pt
		diff = 255
		r,g,b = cv2.split(im)
		r_lo = im[y1,x1,0] - diff
		r_hi = im[y1,x1,0] + diff
		g_lo = im[y1,x1,1] - diff
                g_hi = im[y1,x1,1] + diff
		b_lo = im[y1,x1,2] - diff
                b_hi = im[y1,x1,2] + diff
		color_lo = bw_img[y1,x1] - diff
		color_hi = bw_img[y1,x1] + diff
                phase = int(grad_dir[y1,x1])
		theta_lo = phase - 10
		theta_hi = phase + 10
		#print theta, grad_mag[x1,y1], grad_dir[y1,x1], grad_mag[y1,x1]
		#r = random.randint(0, 255)
		#g = random.randint(0, 255)
		#b = random.randint(0, 255)
		r = 255
		g = 0
		b = 0
		#kernel = np.ones((3,3),np.uint8)
		#closing = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, kernel)
		stroke_swipe = []
		for theta in range(theta_lo, theta_hi):
			for length in range(0,50):
				#theta = theta + deg*(np.pi/180.0)	
				x2 = x1 + length * math.cos(theta)
				y2 = y1 + length * math.sin(theta)
				#debug[y2,x2] = (r,g,b)
				#cv2.imshow("drawing",debug)
				#cv2.imshow("bw_mask", bw_mask)
				#cv2.waitKey(1)	
				#print im[y1,x1,0]
				if y2 < w-1 and x2 < h-1:
					debug[y2,x2] = 100
					cv2.imshow("drawing",debug)
                                	#cv2.imshow("bw_mask", bw_mask)
                                	cv2.waitKey(250) 
					#if im[y2,x2,0] < r_lo or im[y2,x2,0] > r_hi or im[y2,x2,1] < g_lo or im[y2,x2,1] > g_hi or im[y2,x2,2] < b_lo or im[y2,x2,2] > b_hi:
					#if bw_img[y2,x2] < color_lo or bw_img[y2,x2] > color_hi :
					if bw_img[y2,x2] == 0:
						phase_diff = abs(grad_dir[y1,x1] - grad_dir[y2,x2])
						phase_diff = (180.0*phase_diff)/np.pi
						if phase_diff > 95.0 and phase_diff < 275:
							distance = round(np.sqrt((y1-y2)*(y1-y2) + (x1-x2)*(x1-x2)))
							print phase_diff, distance
							stroke_swipe.append(distance)
							print stroke_swipe
						break
				
			#stroke_swipe.append(stroke_wdth)
		s_w = min(stroke_swipe)
		print s_w
		stw_mag.append(s_w)
		
	#print stw_mag
	stw_mag = normalize(stw_mag)
	#print stw_mag
	mu = sum(stw_mag)/float(len(stw_mag))
        stw_array = np.asarray(stw_mag)
        std_stw = np.std(stw_array)
	ratio_check = 1
	if mu > 0:
		ratio_check = std_stw/float(mu)
	#print ratio_check
	return ratio_check

def filter_hiserstic(bw_img):
	#bw_img = gray	
	img = bw_img.copy()	
	filtered_img = np.zeros((h, w), np.uint8)
	contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	
	for cnt in contours:
		#for pt in cnt:
			#print pt
		area = cv2.contourArea(cnt)
		cir_radius = np.sqrt(area/np.pi)
		solidity = solidity_contour(cnt)
		eccentricity = eccentricity_contour(cnt)
		contour_img = np.zeros((h, w), np.uint8)
		#if area > 0:
		if area > 50 and area < 600 and eccentricity < 0.983 and eccentricity > 0.1 and solidity > 0.4:
			#cv2.drawContours(filtered_img,[cnt],-1,255,-1)
			cv2.drawContours(contour_img,[cnt],-1,1,-1)
			contour_mask = cv2.bitwise_and(mser_region_img,contour_img)
			#contour_mask = contour_img*mser_region_img
			contour_img = gray*contour_mask
			#cv2.imshow("contour mask dkdfk" , contour_img)
			#cv2.imshow("mser mask", mser_region_img)
			ratio_check = grad_stw(contour_img, cnt, contour_mask)
			

			dist_img, max_value = stroke_to_width_transform(contour_mask)
			
			r = cv2.boundingRect(cnt)
                	roi = dist_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]	
			mean_img, std_dev_img,std_dev_mean_ratio = mean_std_dev(roi, area)
			print ratio_check, ratio_check, ratio_check, ratio_check, ratio_check
			#if std_dev_mean_ratio < 0.1: #.2
			if ratio_check < 0.5:
				cv2.drawContours(filtered_img,[cnt],-1,255,-1)
			
			#print mean_img, std_dev_img, std_dev_mean_ratio
                	cv2.imshow("contour segment", contour_img)			
			#cv2.waitKey(500)


	return filtered_img

def stroke_to_width_transform(bw_img):
	width_list = []
	float32_img = np.zeros((h, w), np.float32)
	float32_img = bw_img
	dist_img = cv2.distanceTransform( float32_img, cv2.cv.CV_DIST_L2, 3 )
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist_img)
	stroke_Radius = math.ceil(maxVal/2)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	
	for j in range(0,int(stroke_Radius)):
		dist_img = cv2.dilate(dist_img,kernel,iterations = 1)
	dist_img = dist_img*bw_img/255
	norm_dist_img = dist_img
	#cv2.normalize(norm_dist_img, norm_dist_img, 0.0, 1.0, cv2.NORM_MINMAX);
	return norm_dist_img, maxVal

#im   = cv2.imread('tesseract1.jpg')
im   = cv2.imread('text1.png')
#debug = im.copy()
#im   = cv2.imread('text2.png')
h, w = im.shape[:2]
#cv2.imshow('original',im)

im = filter_preprocess(im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 20, 100)
mser_region_img = mser_region_mask(im)
edge_mser_inter = cv2.bitwise_and(edges, mser_region_img)

#TBD gradient growing and mser enahancement
#gradient_grown = grow_edges( gray, edge_mser_inter)

# replace with enhanced image
edge_enhanced_mser = mser_region_img

#filter connected components with hueristics
hFilter_img = filter_hiserstic(edge_enhanced_mser)
hFilter_img = cv2.bitwise_and(hFilter_img, mser_region_img)

# stroke to width transform
distance_img = stroke_to_width_transform(hFilter_img)


#cv2.imshow('mser region', edge_mser_inter)
#cv2.imshow('filtered',hFilter_img)
#cv2.imshow('regions', mser_region_img)
#cv2.imshow('preprocessed', im)
cv2.imshow('final candidtates', hFilter_img)
#cv2.imshow('distance', distance_img)

k = cv2.waitKey(0)
#if k == 27:
#	break

cv2.destroyAllWindows()
