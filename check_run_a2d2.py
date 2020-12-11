### KS created at 2020/12.04
### I use the public repository 'DeepLabV3Plus-Pytorch' https://github.com/VainF/DeepLabV3Plus-Pytorch
#import module
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import models,transforms

sys.path.append('utils')
import extTransform as ex
from lblTransform import transform_seg_label
from make_datapath import make_datapath_list_a2d2
from a2d2_dict import LabelDict
from A2D2Dataset import A2D2Dataset

sys.path.append('DeepLabV3Plus-Pytorch')
import network
from metrics import StreamSegMetrics
sys.path.append('pspnet-pytorch')
from pspnet import PSPNet

#creating image, segmented image list
np.random.seed(1234)
###Setting
## root path
rootpath = '../mmsegmentation/data/a2d2/camera_lidar_semantic'     

##Segmentation label dict (from 
# https://github.com/open-mmlab/mmsegmentation/pull/175/files/722923584878157e7249c331a2c3279099640663)
#It might be need additional transformation such as min -> 0 and max -> the unique number of the values
SEG_COLOR_DICT_A2D2 = LabelDict().dict
cls_num = len(np.unique(list(SEG_COLOR_DICT_A2D2.values())))

##batch size
batch_size=4
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
## model
#from https://github.com/VainF/DeepLabV3Plus-Pytorch
#the model already downloaded pretrained model

model_list = ['deeplabv3_resnet', 'deeplabv3plus_resnet',
'PSPnet resnet18', 'PSPnet resnet34', 'PSPnet resnet50', 
'PSPnet resnet101', 'PSPnet resnet152']
#cannot run all for GPU memory 
model_name = np.random.choice(model_list)
if model_name == "deeplabv3_resnet":
    ##deeplabv3_resnet
    model = network.deeplabv3_resnet50(num_classes=21)
    weight_path = './checkpoints/deeplabv3_r50-d8_513x513_from_source.pth'
    model.load_state_dict(torch.load(weight_path))
    #adjest for the a2d2 class 
    model.classifier.classifier[4] = nn.Conv2d(256, cls_num, kernel_size=(1, 1), stride=(1, 1))
elif model_name == 'deeplabv3plus_resnet':
    ##deeplabv3plus_resnet
    model = network.deeplabv3plus_resnet50(num_classes=21)
    weight_path = './checkpoints/deeplabv3plus_r50-d8_513x513_from_source.pth'
    model.load_state_dict(torch.load(weight_path))
    #adjest for the a2d2 class 
    model.classifier.classifier[3] = nn.Conv2d(256, cls_num, kernel_size=(1, 1), stride=(1, 1))
##PSPNet
elif model_name =='PSPnet resnet34':
    #PSPnet resnet34
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34')
    model = nn.DataParallel(model)
    #adjest for the a2d2 class 
    model.module.classifier[2] = nn.Linear(in_features=256, out_features=cls_num, bias=True)
    model.module.final[0] = nn.Conv2d(64, cls_num, kernel_size=(1, 1), stride=(1, 1))
elif model_name =='PSPnet resnet50':
    #PSPnet resnet50
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50')
    model = nn.DataParallel(model)
    #adjest for the a2d2 class 
    model.module.classifier[2] = nn.Linear(in_features=256, out_features=cls_num, bias=True)
    model.module.final[0] = nn.Conv2d(64, cls_num, kernel_size=(1, 1), stride=(1, 1))
elif model_name =='PSPnet resnet101':
    #PSPnet resnet101
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101')
    model = nn.DataParallel(model)
    #adjest for the a2d2 class 
    model.module.classifier[2] = nn.Linear(in_features=256, out_features=cls_num, bias=True)
    model.module.final[0] = nn.Conv2d(64, cls_num, kernel_size=(1, 1), stride=(1, 1))
elif model_name =='PSPnet resnet152':
    #PSPnet resnet152
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
    model = nn.DataParallel(model)
    #adjest for the a2d2 class 
    model.module.classifier[2] = nn.Linear(in_features=256, out_features=cls_num, bias=True)
    model.module.final[0] = nn.Conv2d(64, cls_num, kernel_size=(1, 1), stride=(1, 1))
elif model_name =='PSPnet resnet18':
    #PSPnet resnet18
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18')
    model = nn.DataParallel(model)
    #adjest for the a2d2 class 
    model.module.classifier[2] = nn.Linear(in_features=256, out_features=cls_num, bias=True)
    model.module.final[0] = nn.Conv2d(64, cls_num, kernel_size=(1, 1), stride=(1, 1))

print(model_name)
#USE GPU
cuda = 0
model.to(cuda)
#Only inference
model.eval()
# cretate path list
train_list, val_list = make_datapath_list_a2d2(rootpath)
#Create Dataset
train_dataset = A2D2Dataset(file_list = train_list, transform=transform_dict['train'], 
                            phase='train',seg_label=SEG_COLOR_DICT_A2D2)
#Create Dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#criterion
metrics = StreamSegMetrics(cls_num)


#run
for image, seg in tqdm(train_dataloader):
    #input image into model
    image = image.cuda(cuda)
    output = model(image)
    
    #set label
    target = seg.to(dtype=torch.long)
    #for pspnet
    if type(output) == tuple:
        output = output[0]
    #back to cpu
    output = output.to('cpu')
    target = target.to('cpu')
    #caluculate loss
    loss = F.cross_entropy(output, target, reduction='mean')
    print(loss.detach())

    #show image
    #label
    plt.imshow(target.detach().numpy()[0])
    plt.savefig('label.png')

    #output
    plt.imshow(output.max(1)[1].detach().numpy()[0].astype(np.uint8))
    plt.savefig(model_name +' output.png')
print('done!')
del model
del image
torch.cuda.empty_cache()


