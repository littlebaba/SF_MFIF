
import os
import torch
from abc import ABC,abstractmethod
from util.util import *
from collections import OrderedDict


class BaseModel(ABC):
    """The class is a abstract base class for submodels.
    
    when creating a subclass,the following funtions need to implement.
        --<__init__>                 initialize the class;first call BaseModel.__init__(self,opt).
        --<set_input>                unpack data from dataset and apply preprocessing.
        --<forward>                  produce intermediate results.
        --<optimize_paramters>       calculate loss,gredients,and update network weights.
    """
    
    def __init__(self,opt):
        """initialize the BaseModel class
        
        When you create a subclass,you should implement your initialization.
        It should first call this funtion.e.g,BaseModel.__init__(self,opt).
        And then you should define the following four lists.
            --self.loss_names (str list)          specify the training losses that you want to plot and save.
            --self.model_names (str list)         define networks used in our training.
            --self.visual_names (str list)        specify the images that you want to show and save.
            --self.opitimizers(opitimizer list)   (optionally) add model-specific options.
        """
        self.opt=opt
        self.isTrain=str2_bool(opt('isTrain'))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir=os.path.join(opt('checkpoints_dir'),opt('name')) 
        self.loss_names=[]
        self.model_names=[]
        self.visual_names=[]
        self.optimizers=[]
        self.image_paths=[]
        self.metric=0
        
    @abstractmethod
    def set_input(self,input):
        """unpack input data from dataloader and perform necessary pre-processing steps.
        
        Paramters:
            input (dic)
        """
        pass
        
    @abstractmethod
    def forward(self):
        """run forward pass."""
        pass
    
    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses,gridents,and unpdate weights;called in every training iteration."""
        pass
    
    def eval(self):
        """Make models eval mode in test time."""
        for name in self.model_names:
            if isinstance(name,str):
                net=getattr(self,'net'+name)
                net.eval()
    
    def test(self):
        """Forward function used in forward."""
        with torch.no_grad():
            self.forward()
    

    def setup(self,opt):
        """Create a scheduler;load and print networks."""
        
        if not self.isTrain:
            load_suffix='iter_%d'%int(self.opt('load_iter'))
            self.load_networks(load_suffix)
        self.print_network(str2_bool(opt('verbose')))
     
    def load_networks(self,epoch):
        """Load network from disk.
        
        Parameters:
            eopch (str) -- used in in the file name '%s_net_%s.pth'%(epoch,name)
        """
        for name in self.model_names:
            if isinstance(name,str):
                load_filename='%s_net_%s.pth'%(epoch,name)
                load_path=os.path.join(self.save_dir,load_filename)
                net=getattr(self,'net'+name)
                print('Loading the model from %s'%load_path)
                state_dict=torch.load(load_path,map_location=str(self.device))
                if hasattr(state_dict,'_metadata'):
                    del state_dict._metadata
                    
                net.load_state_dict(state_dict)
        
        
    def get_currnet_visuals(self):
        """Return visualization images."""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name,str):
                visual_ret[name]=getattr(self,name)
        return visual_ret
    
    def get_current_losses(self):
        """Return training loss.train.py will print out it on console,and save it to a file"""
        losses_ret=OrderedDict()
        for name in self.loss_names:
            if isinstance(name,str):
                losses_ret[name]=float(getattr(self,'loss_'+name))
        return losses_ret
    
    
    def save_networks(self,epoch):
        """Save the networks in disk.
        
        Parameters:
            epoch (int) -- currnet epoch;used in the file name '$s_net_%s.pth'%(epoch,name) 
        """
        for name in self.model_names:
            if isinstance(name,str):
                save_filename='%s_net_%s.pth'%(epoch,name)
                save_path=os.path.join(self.save_dir,save_filename)
                net=getattr(self,'net'+name)
                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda()
    
    def get_image_paths(self):
        return self.image_paths
    
    def print_network(self,verbose):
        """Print the total number of parametors in the network.if verbose,print the architecture of the network.

        Parameters:
            verbose (bool) if print the architecture of the network.
        """
        print('-------Network initialized---------')
        for name in self.model_names:
            if isinstance(name,str):
                net=getattr(self,'net'+name)
                num_params=0
                for param in net.parameters():
                    num_params+=param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parametors: %.3f M' % (name,num_params/1e6))
        print('-----------------------------------')
    
    
    def set_requires_grad(self,nets,requires_grad=False):
        if not isinstance(nets,list):
            nets=[nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad=requires_grad