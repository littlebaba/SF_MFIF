"""This module implements an abstract base class for datasets.

It also contain some common transformation menthod for subclasses,which can be later used"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from abc import ABC,abstractmethod

class BaseDataset(data.Dataset,ABC):
    """This class is a abstract base class for datasets.
    
    When you create a subclass,you need to implement the following four functions:
    --<__init__>:             initialize the subclass,first call BaseDateset.__init__(self,opt).
    --<__len__>:              return the length of dataset.
    --<__getitem__>           get a data point.
    """
    
    def __init__(self,opt):
        """"""
        self.opt=opt
        pass
    
    @abstractmethod
    def __len__(self):
        """return the total number of images in the dataset"""
        return 0;
    
    @abstractmethod
    def __getitem__(self,index):
        """Return a data point
        
        Paramters:
            index--a random integer for data indexing
            
        return:
            a dictionary of data """
        pass
    
    
    
def get_transform(opt,grayscale=True,convert=True,resize=True):
    """supply the transformation list for image.
    
    Parameter:
        grayscale (bool) -- if use normlisation for image.
        convert (bool) -- convert image type from PIL to PyTorch Tensor.
        resize (bool) -- if resize the image. Train (False),Test (True)
    return: a object of transforms.Compose.
    """
    transform_list=[]
    
    if resize:
        transform_list+=[transforms.Resize((256,256)),transforms.Grayscale()]
#         transform_list+=[transforms.CenterCrop(128),transforms.Grayscale()]
    if convert:
        transform_list+=[transforms.ToTensor()] 
        
        if grayscale:
            transform_list+=[transforms.Normalize((0.5,),(0.5,))]
        
    return transforms.Compose(transform_list)
        