# Depth-Segnet

## Segnet using RGBD images

> This is a example of segmentation done using Depth as an additional parameter.  
> Depth is converted into three fields : horizontal disparity, height above ground, and the angle the pixel’s local surface  
normal makes with the inferred gravity direction.  
> This is explained in this paper (https://people.eecs.berkeley.edu/~sgupta/pdf/rcnn-depth.pdf).  

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)  


> This network takes as input a RGB+HHA image and starts training  
> Script to make a lmdb from RGB + HHA + Segmented mask file is provided  
> You can also download the converted lmdb from:  
> Data(RGB+HHA+ Segmented Masks) https://drive.google.com/drive/folders/0B9Zck1VQgjZObG8ySHBuR2JieDQ?usp=sharing  
> The modified prototxts are included in models as (segnet_depth)  
> NYUdv2 dataset was used for training  
> Follow the rest of instructions from  original Segnet.


### References

> SegNet:  http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html  



