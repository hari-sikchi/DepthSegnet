import numpy as np
#import deepdish as dd
import lmdb

import glob
import random
import tifffile
from scipy import misc
import scipy.io
from PIL import Image
datalabel = scipy.io.loadmat('/home/harshit/work/fast-rcnn/data/nyud2/benchmarkData/metadata/classMapping04.mat')


fdata = glob.glob("/home/harshit/work/fast-rcnn/data/nyud2/benchmarkData/groundTruth/*.mat")

for k in range (len(fdata)):
    fcurrent = fdata[k]
    data = scipy.io.loadmat(fcurrent)
    print data['groundTruth'][0][0][0][0][0][1,2]

    for i in range(data['groundTruth'][0][0][0][0][0].shape[0]):
        for j in range(data['groundTruth'][0][0][0][0][0].shape[1]):
            #print datalabel['mapClass'][0][data['groundTruth'][0][0][0][0][0][i,j]]
            data['groundTruth'][0][0][0][0][0][i,j]= datalabel['mapClass'][0,data['groundTruth'][0][0][0][0][0][i,j]]
    print fcurrent
    img = Image.fromarray(data['groundTruth'][0][0][0][0][0], 'L')
    img.save(fcurrent+".png")

#print ((data['groundTruth'][0][0][0][0][0]).shape)
#img = Image.fromarray(data['groundTruth'][0][0][0][0][0], 'L')
#img.save('my.png')
#img.show()
#print (type(data['groundTruth']))
#print data['groundTruth'].shape
#np.copyto(train, data['groundTruth'])
#print train.shape
#ctr=0
#for value in data['groundTruth'].iteritems() :
#    ctr=ctr+1 
#print ctr