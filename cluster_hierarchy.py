import numpy as np
import cv2
from matplotlib import pyplot as plt
import itertools
import os
from PIL import Image
from numpy import *
import hcluster

import scipy
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt


#create a list of images
imlist = []
for filename in os.listdir('./'):
    if os.path.splitext(filename)[1] == '.jpg':
        imlist.append(filename)
n = len(imlist)
#extract feature vector for each image
features = zeros((n,3))
kaze = cv2.KAZE()
for i in range(n):
    img = cv2.imread(imlist[i],0)
    
    #cv2.imshow("img",img)
    #cv2.waitKey(1000)
    kp, des = kaze.detectAndCompute( img, None )
    
    im = array(Image.open(imlist[i]))
    R = mean(im[:,:,0].flatten())
    G = mean(im[:,:,1].flatten())
    B = mean(im[:,:,2].flatten())
    features[i] = array([R,G,B])

tree = hcluster.hcluster(features)

hcluster.drawdendrogram(tree,imlist,jpeg='sunset.jpg')
