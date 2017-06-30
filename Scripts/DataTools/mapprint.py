import numpy as np
#import deepdish as dd
import lmdb

import glob
import random
import tifffile
from scipy import misc
import scipy.io
from PIL import Image

#data = scipy.io.loadmat('/home/harshit/work/fast-rcnn/data/nyud2/benchmarkData/gt_box_cache_dir/img_5001.mat')
data = scipy.io.loadmat('/home/harshit/work/fast-rcnn/data/nyud2/benchmarkData/metadata/classMapping04.mat')

#train= np.zeros_like(data['groundTruth'])
#print data['imnames'][0][5]
print data['allClassName'].shape
#print ((data['groundTruth'][0][0][0][0][0]).shape)
#img = Image.fromarray(data['groundTruth'][0][0][0][0][0], 'L')
#img.save('my.png')
#img.show()
#print (type(data['groundTruth']))
#print data['groundTruth'].shape
#np.copyto(train, data['groundTruth'])
#print train.shape
#ctr=0
for key,value in data.iteritems() :
	print key
    
#print ctr