""""""
from data.base_dataset import get_transform, BaseDataset
from data.image_folder import make_dataset
from util.util import *


class MfDataset(BaseDataset):
    """multi-focus image dataset"""

    def __init__(self, opt):
        # save the option and data root.
        BaseDataset.__init__(self, opt)
        # get image paths of the using dataset.
        self.image_paths = []
        if str2_bool(opt('isTrain')):
            temp = make_dataset(opt('dir'), is_direct_dir=False, isTrain=str2_bool(opt('isTrain')),
                                set_dataset_max=int(opt('set_dataset_size')))
            self.image_paths = temp
            self.transform = get_transform(opt)
        else:
            temp = make_dataset(opt('dir'), is_direct_dir=True, isTrain=str2_bool(opt('isTrain')))
            length = len(temp) // 2
            for i in range(length):
                self.image_paths.append(temp[i * 2:i * 2 + 2])
            # define default transform function.
            self.transform = get_transform(opt)

    def __len__(self):
        """return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """reutrn a data point
        
        Step1:get a random image path. e.g. path=self.image_paths[index].
        Step2:use PIL to read image. e.g. img=Image.open(path).convert('RGB').
        Step3:convert PIL obejct to Tensor. e.g. img=torchvision.transforms.ToTensor(img) 
        Step4:return a data point as a dictionary"""

        if str2_bool(self.opt('isTrain')):
            o = self.image_paths[index]
            l, r, t = blur_image(Image.open(o), 8, style='C')
            target = self.transform(t)
            left = self.transform(l)
            right = self.transform(r)
            return {'left': left, 'right': right, 'target': target, 'l_path': o, 'r_path': o, 't_path': o}
        else:
            l, r = self.image_paths[index]
            left = self.transform(Image.open(l))
            right = self.transform(Image.open(r))
            return {'left': left, 'right': right, 'l_path': l, 'r_path': r}
