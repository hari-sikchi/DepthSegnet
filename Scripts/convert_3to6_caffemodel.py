import numpy as np
import matplotlib.pyplot as plt
import caffe
import numpy as np

caffe.set_mode_gpu()

net = caffe.Net('/home/harshit/Downloads/VGG_ILSVRC_16_layes.prototxt', 
                '/home/harshit/Downloads/VGG_ILSVRC_16_layers.caffemodel', 
                caffe.TEST)


#print type(net.params)
net.params['conv1_1'][0].data=None
print net.params['conv1_1'][0].data.shape

#print ((np.append(net.params['conv1_1'][0].data[0],net.params['conv1_1'][0].data[0],0)).shape)
#z= np.c_(net.params['conv1_1'][0].data,net.params['conv1_1'][0].data[0])
for i in range(net.params['conv1_1'][0].data.shape[0]):
	np.concatenate((net.params['conv1_1'][0].data[i],net.params['conv1_1'][0].data[i]),axis=0)

print net.params['conv1_1'][0].data[0].shape

for key, value in net.params.iteritems() :
    print key
