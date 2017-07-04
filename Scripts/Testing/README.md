

# Scripts for testings


> First compute the BN statistics of your trained model using the compute_bn_statistics script. Remember to make changes to path as given in the script.  

  
> Run test_segmentation_depth.py to see the results. Remember to change the path and categories in test_segmentation_depth.py
  
> [Optional Debugging]Run test_segmentation_depth_filter.py to see the results with filter visualizations. Remember to change the path and categories in test_segmentation_depth_filter.py, also to add the name of layers you want to visualize.
  
### Running Commands  


> python Scripts/test_segmentation_depth.py --model Models/segnet_depth_inference.prototxt --weights Models/Inference/trained_weights.caffemodel --iter 233  