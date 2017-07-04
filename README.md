# Depth-Segnet
  


## Segnet using RGB-D images

> This is a example of segmentation done using Depth as an additional parameter.  
> Depth is converted into three fields : horizontal disparity, height above ground, and the angle the pixel’s local surface  
normal makes with the inferred gravity direction.  
> This is explained in this paper (https://people.eecs.berkeley.edu/~sgupta/pdf/rcnn-depth.pdf).  

![alt text](https://github.com/hari-sikchi/DepthSegnet/blob/master/depthsegnet.png)  


> This network takes as input a RGB+HHA image and starts training  
> Script to make a lmdb from RGB + HHA + Segmented mask file is provided  
> You can also download the converted lmdb from:  
> Data(RGB+HHA+ Segmented Masks)[link](https://drive.google.com/open?id=0B9Zck1VQgjZOXzh2RXdsMFVLV1E)    
> The modified prototxts are included in models as (segnet_depth)  
> NYUdv2 dataset was used for training  
> Follow the rest of instructions from  original Segnet.
  
  

> Modified scripts can be found in datatools folder.    

### Examples

> Segmentation results in cluttered scenes from NYUDv2 dataset [Added Filter output visualization].
![alt text](https://github.com/hari-sikchi/DepthSegnet/blob/master/Filtervisualization.png)
>----------------------------------------------------------------------------------------------      
![alt text](https://github.com/hari-sikchi/DepthSegnet/blob/master/cluttered_segmentation1.png)
>----------------------------------------------------------------------------------------------      
![alt text](https://github.com/hari-sikchi/DepthSegnet/blob/master/clutteredsegmentation2.png)    
>----------------------------------------------------------------------------------------------      
![alt text](https://github.com/hari-sikchi/DepthSegnet/blob/master/clutteredsegmentation3.png)    


> Final trained weights can be found here. [link](https://drive.google.com/open?id=0B9Zck1VQgjZObzJVTkVpYVZ6bVk)
### References

> SegNet:  http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html  





