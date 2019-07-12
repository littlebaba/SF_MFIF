
import numpy as np
import torch
from PIL import Image,ImageFilter
import os

def str2_bool(str):
    """Convert a str to bool"""
    return True if str.lower() =='true' else False

def tensor2im(input_image,imtype=np.uint8):
    """Convert a Tensor array into a numpy array.
    
    Parameters:
        input_image (Tensor) -- the input image of Tensor array.
        imtype (type) -- the desired type of the converted image array.
    return a numpy array with imtype
    """
    assert isinstance(input_image,torch.Tensor),'Error: the input type is not torch.Tensor.'
    image_numpy=input_image.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy=np.tile(image_numpy,(3,1,1))
    image_numpy=(np.transpose(image_numpy,(1,2,0))+1)/2*255.0
    return image_numpy.astype(imtype)
    
    
def save_image(image_numpy,image_path):
    """Save image to disk.
    
    Paramters:
        image_numpy (ndarray) -- the image need to save.
        image_path (str) -- the path of saved image.
    """
    image_pil=Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
def mkdirs(paths):
    """construct directories.
    
    Parameter:
        paths (str list) -- the str list that need to create directories.
    """
    if isinstance(paths,list):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
    
def mkdir(path):
    """create directory if it not exist.
    
    Paramters:
        path (str) -- the directory path.
    """
    if not os.path.exists(path):
        os.mkdir(path)
        
def blur_image(image,num_div,style='C'):
    """Blur source image and generate two blured images.And the two images are complementary.
    Parameters:
        image (object) -- source image that the type is PIL object;it should be equal in length and width，and its all should all even.
        num_div (int) -- the number of times that source image id divided.
        style (str) -- blurred style;'P':parallel and 'C':cross.
    Returns:
        triple-tulpe (l,r,o).It is left focus image,right focus image,original image in turn.     
    """
    l=Image.new('L',image.size)
    r=Image.new('L',image.size)
    blocks=[]
    block_size=image.size[0]//num_div
    for row in range(1,num_div+1):
        for col in range (1,num_div+1):
            box=(block_size*(col-1),block_size*(row-1),block_size*(col-1)+block_size+1,block_size*(row-1)+block_size+1)
            block=image.crop(box)
            blocks.append(block)
            blur_block=block.filter(ImageFilter.GaussianBlur(radius=3))
            if style=='C':
                if row%2==1:
                    if (num_div*(row-1)+(col-1))%2==0:
                        l.paste(blur_block,box)
                        r.paste(block,box)
                    else:
                        l.paste(block,box)
                        r.paste(blur_block,box)
                else:
                    if (num_div*(row-1)+(col-1))%2==1:
                        l.paste(blur_block,box)
                        r.paste(block,box)
                    else:
                        l.paste(block,box)
                        r.paste(blur_block,box)
            elif style=='P':
                    if (num_div*(row-1)+(col-1))%2==0:
                        l.paste(blur_block,box)
                        r.paste(block,box)
                    else:
                        l.paste(block,box)
                        r.paste(blur_block,box)
    return l,r,image