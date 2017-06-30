import numpy as np
import matplotlib.pyplot as plt
import caffe
import numpy as np

caffe.set_mode_gpu()

net = caffe.Net('/home/harshit/work/SegNet-Tutorial/Models/VGG16_Train.prototxt', 
                '/home/harshit/work/SegNet-Tutorial/Models/Training/segnet0_iter_5.caffemodel', 
                caffe.TRAIN)


net2= caffe.Net('/home/harshit/Downloads/VGG_ILSVRC_16_layes.prototxt','/home/harshit/Downloads/VGG_ILSVRC_16_layers.caffemodel',caffe.TEST)


for key, value in net.params.iteritems() :
    print type(net.params[key][1].data)

for key,value in net.params.iteritems():
	if(key == 'conv1_1'):
		np.copyto(net.params[key][1].data,net2.params[key][1].data)
		for i in range(net2.params['conv1_1'][0].data.shape[0]):
			np.copyto(net.params['conv1_1'][0].data[i],np.concatenate((net2.params['conv1_1'][0].data[i],net2.params['conv1_1'][0].data[i]),axis=0))

	elif(key== 'fc6' or key== 'fc7'):
		continue							
	else:

		np.copyto(net.params[key][0].data,net2.params[key][0].data)
		np.copyto(net.params[key][1].data,net2.params[key][1].data)



net.save('bvlc_caffenet_full_RGBD.caffemodel')
#print type(net.params)

#print net.params['conv1_1'][0].data.shape
#print net2.params['conv1_1'][0].data.shape


#print ((np.append(net.params['conv1_1'][0].data[0],net.params['conv1_1'][0].data[0],0)).shape)
#z= np.c_(net.params['conv1_1'][0].data,net.params['conv1_1'][0].data[0])
#for i in range(net.params['conv1_1'][0].data.shape[0]):
#	np.concatenate((net.params['conv1_1'][0].data[i],net.params['conv1_1'][0].data[i]),axis=0)

#print net.params['conv1_1'][0].data[0].shape

#for key, value in net.params.iteritems() :
#    print key
