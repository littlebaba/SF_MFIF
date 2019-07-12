import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from util.util import str2_bool

def find_dataset_using_name(dataset_name):
    dataset_filename = "data."+dataset_name + "_dataset"
    datasetlib=importlib.import_module(dataset_filename)
    target_dataset_name=dataset_name.replace('_','')+'dataset'
    dataset=None
    for name,cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls,BaseDataset):
            dataset=cls
    return dataset

def create_dataset(opt):
    data_loader=CustomDatasetDataLoader(opt)
    return data_loader.load_data()

class CustomDatasetDataLoader():
    def __init__(self,opt):
        self.opt=opt
        dataset_class=find_dataset_using_name(opt('dataset_mode'))
        self.dataset=dataset_class(opt)
        print('dataset [%s] was created' % type(self.dataset).__name__)
        self.dataloader=torch.utils.data.DataLoader(
            self.dataset,batch_size=int(opt('batch_size')),
            shuffle=str2_bool(opt('suffle_batches')),num_workers=int(opt('num_threads')))
    
    def load_data(self):
        return self
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """return a batch of data."""
        for i,data in enumerate(self.dataloader):
            yield data