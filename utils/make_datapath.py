import os 
import glob
import numpy as np

def make_datapath_list_a2d2(rootpath, train_rate= 1.0):
    """
    Create list that contains the path for train/inference model

    Parameters 
    rootpath: str
        the path for data folder 
    split rate: float ranged 0.0 to 1.0 (default 1.0)
        the rate to split the path for train data in rootpath 

    Returns
    train_id_names: list of list
        the path for training data in a2d2 dataset [[image], [segmentation]]
    eval_id_names: list of list 
        tha path for evaluation data in a2d2 dataset
    
    """
    #check train_rate
    if train_rate < 0 or 1 < train_rate:
        raise ValueError("The train rate should be within 0 to 1.")

    #Create path temporale for image (and segmentated image)
    imgpath_template = np.array(glob.glob(os.path.join(rootpath,'*','camera', '*', '*.png' )))
    segimgpath_template = np.array(glob.glob(os.path.join(rootpath,'*','label', '*', '*.png' )))
    
    #split train and test
    train_id = np.random.choice(np.arange(len(imgpath_template)),int(len(imgpath_template)* train_rate), replace = False )
    
    train_id_names = [imgpath_template[train_id], segimgpath_template[train_id]]
    # the evaluation path is selected from the path that not contains in train_id_names
    eval_id_names = [np.array([p for p in imgpath_template if p not in train_id_names[0]]),
                     np.array([p for p in segimgpath_template if p not in train_id_names[1]])]
    return train_id_names, eval_id_names