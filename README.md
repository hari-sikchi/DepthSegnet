# Depth-Segnet
  


## Encoder-Decoder semantic segmentation with depth fusion using RGB-D images

> Segmentation done using Depth as an additional parameter.  
> Depth is converted into three fields : horizontal disparity, height above ground, and the angle the pixel’s local surface  
normal makes with the inferred gravity direction.  
> Idea taken from paper [Depth-RCNN](https://people.eecs.berkeley.edu/~sgupta/pdf/rcnn-depth.pdf).  

<img src="https://github.com/hari-sikchi/DepthSegnet/blob/master/depthsegnet.png" width="1000">


> This network takes as input a RGB+HHA image[6-channels].  
> Script to make a lmdb from RGB + HHA + Segmented mask file is provided.  
> You can also download the converted lmdb from:  
> Data(RGB+HHA+ Segmented Masks)[link](https://drive.google.com/open?id=0B9Zck1VQgjZOXzh2RXdsMFVLV1E)    
> The modified prototxts are included in models as (segnet_depth_*).  
> NYUdv2 dataset was used for training.
> Follow the rest of instructions from  original Segnet.
  
  

> Modified scripts can be found in datatools folder.    

## Setup

1. Clone this repository: `git clone https://github.com/hari-sikchi/DepthSegnet.git`    
2. Initialize all submodules: `git submodule update --init --recursive`  
3. Download all the data from the link given below or create your own data in the format:  
       Data/data_lmdb(Containing images of 6 channel)  
       Data/labels(Containing segmentation masks for each channel)
4. Start training and see for yourself!         

### Examples

> Segmentation results in cluttered scenes from NYUDv2 dataset [Added Filter output visualization].
![alt text](https://github.com/hari-sikchi/DepthSegnet/blob/master/Filtervisualization.png)
>----------------------------------------------------------------------------------------------      
<img src="https://github.com/hari-sikchi/DepthSegnet/blob/master/cluttered_segmentation1.png" width="288"> <img src="https://github.com/hari-sikchi/DepthSegnet/blob/master/clutteredsegmentation2.png" width="288"> <img src="https://github.com/hari-sikchi/DepthSegnet/blob/master/clutteredsegmentation3.png" width="288">   


> Final trained weights can be found here. [Link](https://drive.google.com/open?id=0B9Zck1VQgjZObzJVTkVpYVZ6bVk)
### References

> SegNet:  http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html  





