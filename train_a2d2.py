### KS created at 2020/12.04
### I use the public repository 'DeepLabV3Plus-Pytorch' https://github.com/VainF/DeepLabV3Plus-Pytorch

#import module
import glob
import os
import sys
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms

import extTransform as ex
from lblTransform import transform_seg_label

sys.path.append('DeepLabV3Plus-Pytorch')
import network
from metrics import StreamSegMetrics

#creating image, segmented image list
np.random.seed(123)
def make_datapath_list(rootpath, split_rate= 1.0):
    """
    Create list that contains the path for train/inference model

    Parameters 
    rootpath: str
        the path for data folder 
    split rate: float ranged 0.0 to 1.0 (default 1.0)
        the rate to split the path for train data in rootpath 

    Returns
    train_id_names: str
        the path for training data in a2d2 dataset
    eval_id_names: str
        tha path for evaluation data in a2d2 dataset
    
    """
    #Create path temporale for image (and segmentated image)
    imgpath_template = np.array(glob.glob(os.path.join(rootpath,'*','camera', '*', '*.png' )))
    segimgpath_template = np.array(glob.glob(os.path.join(rootpath,'*','label', '*', '*.png' )))
    
    #split train and test
    train_id = np.random.choice(np.arange(len(imgpath_template)),int(len(imgpath_template)* split_rate), replace = False )
    
    train_id_names = [imgpath_template[train_id], segimgpath_template[train_id]]
    # the evaluation path is selected from the path that not contains in train_id_names
    eval_id_names = [np.array([p for p in imgpath_template if p not in train_id_names[0]]),
                     np.array([p for p in segimgpath_template if p not in train_id_names[1]])]
    return train_id_names, eval_id_names

class a2d2Dataset(data.Dataset):
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
        return len(self.file_list)
    
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

# create Data (image and segmentated image) Transformer
#Train 
train_transform = ex.ExtCompose([
            ex.ExtRandomCrop(size=(512, 512)),
            ex.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            ex.ExtRandomHorizontalFlip(),
            ex.ExtToTensor(),
            ex.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
#Validation
val_transform = ex.ExtCompose([
    ex.ExtResize( 512 ),
    ex.ExtToTensor(),
    ex.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])
#Dictiorany for easy use
transform_dict ={'train': train_transform,
                'val': val_transform}

###Setting     

##Segmentation label dict (from 
# https://github.com/open-mmlab/mmsegmentation/pull/175/files/722923584878157e7249c331a2c3279099640663)
#It might be need additional transformation such as min -> 0 and max -> the unique number of the values
SEG_COLOR_DICT_A2D2 = {
(255, 0, 0): 28, # Car 1
(200, 0, 0): 28, # Car 2
(150, 0, 0): 28, # Car 3
(128, 0, 0): 28, # Car 4
(182, 89, 6): 27, # Bicycle 1
(150, 50, 4): 27, # Bicycle 2
(90, 30, 1): 27, # Bicycle 3
(90, 30, 30): 27, # Bicycle 4
(204, 153, 255): 26, # Pedestrian 1
(189, 73, 155): 26, # Pedestrian 2
(239, 89, 191): 26, # Pedestrian 3
(255, 128, 0): 30, # Truck 1
(200, 128, 0): 30, # Truck 2
(150, 128, 0): 30, # Truck 3
(0, 255, 0): 32, # Small vehicles 1
(0, 200, 0): 32, # Small vehicles 2
(0, 150, 0): 32, # Small vehicles 3
(0, 128, 255): 19, # Traffic signal 1
(30, 28, 158): 19, # Traffic signal 2
(60, 28, 100): 19, # Traffic signal 3
(0, 255, 255): 20, # Traffic sign 1
(30, 220, 220): 20, # Traffic sign 2
(60, 157, 199): 20, # Traffic sign 3
(255, 255, 0): 29, # Utility vehicle 1
(255, 255, 200): 29, # Utility vehicle 2
(233, 100, 0): 16, # Sidebars
(110, 110, 0): 12, # Speed bumper
(128, 128, 0): 14, # Curbstone
(255, 193, 37): 6, # Solid line
(64, 0, 64): 22, # Irrelevant signs
(185, 122, 87): 17, # Road blocks
(0, 0, 100): 31, # Tractor
(139, 99, 108): 1, # Non-drivable street
(210, 50, 115): 8, # Zebra crossing
(255, 0, 128): 34, # Obstacles / trash
(255, 246, 143): 18, # Poles
(150, 0, 150): 2, # RD restricted area
(204, 255, 153): 33, # Animals
(238, 162, 173): 9, # Grid structure
(33, 44, 177): 21, # Signal corpus
(180, 50, 180): 3, # Drivable cobblestone
(255, 70, 185): 23, # Electronic traffic
(238, 233, 191): 4, # Slow drive area
(147, 253, 194): 24, # Nature object
(150, 150, 200): 5, # Parking area
(180, 150, 200): 13, # Sidewalk
(72, 209, 204): 255, # Ego car
(200, 125, 210): 11, # Painted driv. instr.
(159, 121, 238): 10, # Traffic guide obj.
(128, 0, 255): 7, # Dashed line
(255, 0, 255): 0, # RD normal street
(135, 206, 255): 25, # Sky
(241, 230, 255): 15, # Buildings
(96, 69, 143): 255, # Blurred area
(53, 46, 82): 255, # Rain dirt
}

## root path
rootpath = '../data/a2d2/camera_lidar_semantic'
##batch size
batch_size=2

## model
#from https://github.com/VainF/DeepLabV3Plus-Pytorch
#the model already downloaded pretrained model
cls_num = len(np.unique(list(SEG_COLOR_DICT_A2D2.values())))
#deeplabv3_resnet
model = network.deeplabv3_resnet50(num_classes=21)
weight_path = '../checkpoints/deeplabv3plus_r50-d8_513x513_from_source.pth'
model.load_state_dict(torch.load(weight_path))
model.classifier.classifier[4] = nn.Conv2d(256, cls_num, kernel_size=(1, 1), stride=(1, 1))
# cretate path list
train_list, val_list = make_datapath_list(rootpath)
#Create Dataset
train_dataset = a2d2Dataset(file_list = train_list, transform=transform_dict['train'], 
                            phase='train',seg_label=SEG_COLOR_DICT_A2D2)
#Create Dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#criterion
metrics = StreamSegMetrics(cls_num)


#run
for image, seg in train_dataloader:
    #input image into model
    output = model(image)
    #set label
    target = seg.to(dtype=torch.long)
    #caluculate loss
    loss = F.cross_entropy(output, target, reduction='mean')
    print(loss.detach())

print('done!')
