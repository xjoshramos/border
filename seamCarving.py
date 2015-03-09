import cv2
import numpy as np
import random
import operator


def random_walk(energy):
	path = []
	height, width = energy.shape[:2]
	x = random.randint(0, width-1)
	val = energy[0,x]
	
	path.append((0,x))	
	
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
		path.append((i,x))
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

	while iter_random_path < 200:
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
	crop = im
	for point in path:
		
	#crop = im

	#for point in path:
        #	im[point] = (0,255,0)

	#cv2.imshow('min path',im)
	#cv2.imshow('grad',energy)

	return crop

im = cv2.imread('bordered5.jpg')
#cv2.imshow('houghlines',im)
cropped = im
iter_crop = 0
while iter_crop < 10:

	cropped = crop(cropped)
	iter_crop += 1

k = cv2.waitKey(0)
    #if k == 27:
       # break

cv2.destroyAllWindows()
