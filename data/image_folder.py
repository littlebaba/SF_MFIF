import os
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif','.gif',
]

def is_image_file(file_name):
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir,is_direct_dir,isTrain,set_dataset_max=float('inf')):
    """
        
    Paramters:
        dir (str) -- the folder of images
        is_direct_dir (bool) -- if only choice father dir
    return:
        image (list) -- a list of containing image paths
    """
    
    image=[]
    assert os.path.isdir(dir),'[%s] is not a valid directory.'%dir
    for root,dirs,files in os.walk(dir):
        for file in files:
            if is_image_file(file):
                path=os.path.join(root,file)
                image.append(path)
        if is_direct_dir:
            break
    if isTrain:
        temp=sorted(image,key=lambda x:(int(re.findall(r'\d+',x.split('/')[-1])[0]) ,x.split('/')[-1].split('_')[-1]))
    else:
        temp=sorted(image) #测试数据为dir=E:\图像数据库\多聚焦数据库 时调用
        # temp=sorted(image,key=lambda x:(int(re.findall(r'\d+',x.split('/')[-1])[0]) ,x.split('/')[-1].split('_')[-1]))#测试数据为dir=E:\matPro\duishuanfangduojujiao\original时调用
    return temp[:min(set_dataset_max,len(temp))]