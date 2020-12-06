import numpy as np
from PIL import Image
from itertools import product

import torch
import torch.utils.data as data

class A2D2Dataset(data.Dataset):
    """
    Class of loading a2d2 dataset 

    Attributes
    file_list: list of list
        list containing image and segmetation path
    seg_label: dict
        key... pix RGB, values... the label
    transform: instance (default None)
        the instnace of preprocessing 
    phase: 'train' or 'test'

    """
    
    def __init__(self, file_list, seg_label, transform = None,phase='train'):
        self.file_list = file_list
        self.seg_label = seg_label
        self.transform = transform
        self.phase = phase
        
        
    def __len__(self):
        """
        return the length of image
        """
        return len(self.file_list[0])
    
    def __getitem__(self, index):
        """
        obtrain data and label. data is preprocessed in this func
        """
        
        #load index image and segmentated image
        img_path, lbl_path = self.file_list[0][index], self.file_list[1][index]
        img = Image.open(img_path)
        lbl = np.array(Image.open(lbl_path))
        #convert label
        lbl = transform_seg_label(lbl, self.seg_label)
        #preprocessing
        img_transformed = self.transform(img, lbl)
        
        return img_transformed

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