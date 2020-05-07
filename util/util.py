import os

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms


def str2_bool(str):
    """Convert a str to bool"""
    return True if str.lower() == 'true' else False


def tensor2im(input_image, imtype=np.uint8):
    img = input_image.cpu().detach()[0]
    image_numpy = np.array(img)
    return img,image_numpy


def save_image(image_tensor, image_path):
    """Save image to disk.
    
    Paramters:
        image_numpy (ndarray) -- the image need to save.
        image_path (str) -- the path of saved image.
    """
    trans = transforms.ToPILImage()
    image_pil = trans(image_tensor)
    image_pil.save(image_path)


def mkdirs(paths):
    """construct directories.
    
    Parameter:
        paths (str list) -- the str list that need to create directories.
    """
    if isinstance(paths, list):
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


def blur_image(image, num_div, style='C'):
    """Blur source image and generate two blured images.And the two images are complementary.
    Parameters:
        image (object) -- source image that the type is PIL object;it should be equal in length and width，and its all should all even.
        num_div (int) -- the number of times that source image id divided.
        style (str) -- blurred style;'P':parallel and 'C':cross.
    Returns:
        triple-tulpe (l,r,o).It is left focus image,right focus image,original image in turn.     
    """
    l = Image.new('RGB', image.size)
    r = Image.new('RGB', image.size)
    blocks = []
    block_size = image.size[0] // num_div
    for row in range(1, num_div + 1):
        for col in range(1, num_div + 1):
            box = (block_size * (col - 1), block_size * (row - 1), block_size * (col - 1) + block_size + 1,
                   block_size * (row - 1) + block_size + 1)
            block = image.crop(box)
            blocks.append(block)
            blur_block = block.filter(ImageFilter.GaussianBlur(radius=3))
            if style == 'C':
                if row % 2 == 1:
                    if (num_div * (row - 1) + (col - 1)) % 2 == 0:
                        l.paste(blur_block, box)
                        r.paste(block, box)
                    else:
                        l.paste(block, box)
                        r.paste(blur_block, box)
                else:
                    if (num_div * (row - 1) + (col - 1)) % 2 == 1:
                        l.paste(blur_block, box)
                        r.paste(block, box)
                    else:
                        l.paste(block, box)
                        r.paste(blur_block, box)
            elif style == 'P':
                if (num_div * (row - 1) + (col - 1)) % 2 == 0:
                    l.paste(blur_block, box)
                    r.paste(block, box)
                else:
                    l.paste(block, box)
                    r.paste(blur_block, box)
    return l, r, image
