#
#
# Harshit Sikchi
# RGBD Image segmentation using Segnet
# Data tools
#

import numpy as np
import lmdb
import caffe
import glob
import random
from scipy import misc

fdataRGB = glob.glob("/home/harshit/work/fast-rcnn/data/nyud2/data/images/*.png") #Change the path here to point to your RGB images
fdataHHA=glob.glob("/home/harshit/work/fast-rcnn/data/nyud2/data/hha/*.png") #Change the path here to point to ur HHA images
flabel = glob.glob("/home/harshit/work/fast-rcnn/data/nyud2/benchmarkData/groundTruth/*.png") # Change the path to point to your segmentation masks

fdataRGB =sorted(fdataRGB)
fdataHHA = sorted(fdataHHA)
flabel = sorted(flabel)


n_data = len(fdataRGB)
print(n_data)
final_data = np.zeros((n_data,256,256,6))



for k in range (len(fdataRGB)):
    fcurrent = fdataRGB[k]
    img = misc.imread(fcurrent)
    img=  misc.imresize(img,(256,256,3))
    img = img.reshape(1,256,256,3)   
    final_data[k,:,:,0:3] = img

for k in range (len(fdataHHA)):
    fcurrent = fdataHHA[k]
    img = misc.imread(fcurrent)
    img = misc.imresize(img, (256,256,3))
    img = img.reshape(1,256,256,3)   
    final_data[k,:,:,3:6] = img

print("done") 


#creates a numpy array data
#input format image,channel,width,height

final_data = np.transpose(final_data,(0,3,1,2))
data = final_data
print 'data shape : ',data.shape

#creates a numpy array of label Y

n_label = len(flabel)
final_label = np.zeros((n_label,256,256,1))

for k in range (len(flabel)):
    fcurrent = flabel[k]
    img = misc.imread(fcurrent)
    img = misc.imresize(img, (256,256,1))
    img = img.reshape(1,256,256,1)   
    final_label[k] = img

#input format image,channel,width,height
final_label = np.transpose(final_label,(0,3,1,2))    
Y =  final_label
print 'label shape : ',Y.shape


#shuffle
N = range(len(final_data))
random.shuffle(N)


map_size = Y.nbytes*2
env = lmdb.open('labels', map_size=map_size)
for z in N:
    i = N[z]
    im_dat = caffe.io.array_to_datum(np.array(Y[i]).astype(float))
    str_id = '{:0>10d}'.format(i)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())


print 'Creating image dataset.'
X = np.array(data)
X = np.array(X,dtype=np.float32)
map_size = X.nbytes*2
env = lmdb.open('data_lmdb', map_size=map_size)
for z in N:
    i = N[z]
    im_dat = caffe.io.array_to_datum(np.array(X[i]).astype(float))
    str_id = '{:0>10d}'.format(i)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())


