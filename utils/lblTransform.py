### KS created at 2020/12.04
import numpy as np
from itertools import product
from PIL import Image

def transform_seg_label(lbl, SEG_dict):
    """
    Create image segmentation label from label image 
    which indicates label in [R, G, B] values.

    Parameters 
    lbl: label of image segmentation (Image), Size(height, width, 3)
    SEG_dict: Dictionary whose keys are the tuple indicates [R, G, B] values, 
        and whose values are the assinged label (value) such as 0 (Car1), 1 (Car2),  

    Returns
    label: Image
        assgined label, Size(height, width)
    """
    ## Can we speed up?

    #convert ndarray
    lbl = np.array(lbl)
    #obtain image size
    lbl_shape = lbl.shape[:2]
    #crete empty array for assgining dict values
    label_ = np.zeros_like(lbl[...,0])
    #assign each pixel from coresponding label
    for i, j in product(range(lbl_shape[0]), range(lbl_shape[1])):
        label_[i,j] = SEG_dict[tuple(lbl[i,j])]
    # Return as Image type
    return Image.fromarray(label_)
                       
                        