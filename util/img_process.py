import os
from os.path import join

from PIL import Image


def crop_save2000():
    root = r'E:\data\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
    save = r'E:\图像数据库\bigmfdataset\V1expand_original_images_RGB'
    imgs = [join(root, img_path) for img_path in os.listdir(root)[0:2000]]
    for i, img in enumerate(imgs):
        img = Image.open(img)
        w, h = img.size
        box = (w/2 - 128, h/2 - 128, w/2 + 128, h/2 + 128)
        new_img = img.crop(box)
        new_img.save(join(save, f'image{i + 1}.png'))

if __name__ == '__main__':
    crop_save2000()