import cv2
import numpy as np
import random
import operator


def random_walk(energy):
	path = []
	height, width = energy.shape[:2]
	x = random.randint(0, width-1)
	val = 0 #energy[0,x]
	
	path.append(x)	
	
	energy_total = val
	for i in range(1,height-1):
		if x > 0 and x < width-1:	
			neighbors = [x-1, x, x+1]
			energy_neighbors = [energy[i,x-1], energy[i,x], energy[i,x+1]]
			min_index, min_value = min(enumerate(energy_neighbors), key=operator.itemgetter(1))
		 	
			min_energy = []		
			for z,val in enumerate(energy_neighbors):
				if val == min_value:			
					min_energy.append(neighbors[z])

			x = random.choice(min_energy)
			#print min_index, min_value, energy_neighbors, neighbors[min_index], neighbors, min_energy
		#draw path in blue
		#im[i,x] = (255,0,0)
		path.append(x)
		energy_total += energy[i,x]


	#print path, energy_total
	#cv2.imshow("walk", im)
	
	return path, energy_total

def image_energy(im):
	height, width = im.shape[:2]

	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
	sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

	abs_grad_y = cv2.convertScaleAbs(sobel_y);
	abs_grad_x = cv2.convertScaleAbs(sobel_x);

	energy = cv2.addWeighted(abs_grad_y, 0.5, abs_grad_x, 0.5, 0)

	return energy

def min_path(energy):

	path = []
	iter_random_path = 0

	while iter_random_path < 50:
        	if not path:
                	path, energy_tot = random_walk(energy)
        	else:
                	next_path, next_energy_tot = random_walk(energy)
                	if energy_tot > next_energy_tot:
                        	path = next_path
                        	energy_tot = next_energy_tot
                	#print next_energy_tot
        	#print path, energy_tot
        	iter_random_path += 1
	
	return path


def crop(im):
	energy = image_energy(im)
	path = min_path(energy)
	height, width = im.shape[:2]
	crop = np.zeros((height,width-1,3), np.uint8)
	print width, height, len(path)
	for h in range(0,height-1):
		for w in range(0,width-1):
			#print path[h], len(path)	
			#print pt_w, pt_h
			if w < path[h]:
				#print path[y], 
				crop[h,w] = im[h,w]
			if w >= path[h]:
				crop[h,w] = im[h,w+1]
			#if w == path[h]:
				#cl = im[h,w]
				#cr = im[h,w+1]
				#cn =( (cl[0] + cr[0])/2, (cl[1] + cr[1])/2, (cl[2] + cr[2])/2 )
				#crop[h,w] = cn
				
			#else:
				
	#for point in path:
		#pth, ptw = point
        	#crop[pth,ptw] = (0,255,0)

	#cv2.imshow('min path',im)
	#cv2.imshow('grad',energy)
	#cv2.imshow('crop',crop)

	return crop

im = cv2.imread('seamCarve1.jpg')
cv2.imshow('original',im)
height, width = im.shape[:2]
cropped = im
iter_crop = 0
while iter_crop < 200:

	cropped = crop(cropped)
	
	cv2.imshow('crop',cropped)
	iter_crop += 1

k = cv2.waitKey(0)
    #if k == 27:
       # break

cv2.destroyAllWindows()
