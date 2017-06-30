import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
caffe_root = '/home/harshit/work/SegNet-Tutorial/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()

	C1 = [0,0,20]
	C2 = [0,0,40]                         #### Declare the colours for each category you have in your dataset.
	C3 = [120,0,0]
	C4 = [0,0,80]
	C5 = [0,0,100]
	C6 = [60,0,0]
	C7 = [0,0,140]
	C8 = [200,0,0]
	C9 = [0,0,240]
	C10 = [0,20,0]
	C11 = [0,40,0]
	C12 = [0,60,0]
	C13 = [20,0,0]
	C14 = [0,100,0]
	C15 = [140,0,0]
	C16 = [0,140,0]
	C17 = [0,200,0]
	C18 = [0,240,0]
	C19 = [0,80,0]
	C20 = [40,0,0]
	C21 = [0,0,120]
	C22 = [80,0,0]
	C23 = [100,0,0]
	C24 = [0,0,60]
	C25 = [0,120,0]
	C26 = [0,0,200]
	C27 = [240,0,0]


				# Change this array according to number of categories
	label_colours = np.array([C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27])


	for l in range(0,27):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]
		r_gt[label==l] = label_colours[l,0]
		g_gt[label==l] = label_colours[l,1]
		b_gt[label==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = r_gt/255.0
	rgb_gt[:,:,1] = g_gt/255.0
	rgb_gt[:,:,2] = b_gt/255.0

	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]


	

	plt.figure()
	plt.imshow(image,vmin=0, vmax=1)
	plt.figure()
	plt.imshow(rgb_gt,vmin=0, vmax=1)
	plt.figure()
	plt.imshow(rgb,vmin=0, vmax=1)
	plt.show()
	#plt.imsave('/home/sharshit/Segnet/result.png',rgb)


print 'Success!'

