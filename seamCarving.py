import cv2
import numpy as np
import random
import operator


def random_walk(energy):
	path = []
	#print height, width
	x = random.randint(0, width-1)
	#print x
	val = energy[0,x]
	
	#path.append((x,y,val))
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
		im[i,x] = (255,0,0)
		path.append((i,x))
		energy_total += energy[i,x]


	#print path, energy_total
	cv2.imshow("walk", im)
	





	return path, energy_total


im = cv2.imread('bordered5.jpg')
cv2.imshow('houghlines',im)
height, width = im.shape[:2]

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

abs_grad_y = cv2.convertScaleAbs(sobel_y);
abs_grad_x = cv2.convertScaleAbs(sobel_x);

energy = cv2.addWeighted(abs_grad_y, 0.5, abs_grad_x, 0.5, 0)


k = 0
while k<(2):
	k += 1
	path, energy_tot = random_walk(energy)
	print path, energy_tot

cv2.imshow('grad',energy)

k = cv2.waitKey(0)
    #if k == 27:
       # break

cv2.destroyAllWindows()
