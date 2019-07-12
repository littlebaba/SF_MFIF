import torch
from models.base_model import BaseModel
from . import networks

class MfModel(BaseModel):
    """This class implement multi focus model for learning mapping from a pair of focus images to a source image.
    
    """
    
    def __init__(self,opt):
        """initialize the multi focus class."""
        BaseModel.__init__(self,opt)
        
        self.loss_names=['Mf']
        self.model_names=['Mf']
        
        if self.isTrain:
            self.visual_names=['left','right','target','output']
        else:
            self.visual_names=['left','right','output']
        self.netMf=networks.define_Mf(netMf=opt('netMf'),device=self.device)
        
        if self.isTrain:
            #self.critirion=networks.MfLoss()
            #self.critirion=torch.nn.L1Loss()
            self.critirion1=networks.SSIMLoss().to(self.device)
            self.critirion2=torch.nn.MSELoss()
            self.optimizer_Mf=torch.optim.Adam(self.netMf.parameters(),lr=float(opt('lr')),betas=(float(opt('beta1')),0.999))
            self.optimizers.append(self.optimizer_Mf)
            
    
    def set_input(self,input):
        """unpack data from dataloader and perform necessary pre-processing steps.
        
        """
        if self.isTrain:
            self.left=input['left'].to(self.device)
            self.right=input['right'].to(self.device)
            self.target=input['target'].to(self.device)
            self.image_paths=[input['l_path'],input['r_path'],input['t_path']]
        else:
            self.left=input['left'].to(self.device)
            self.right=input['right'].to(self.device)
            self.image_paths=[input['l_path'],input['r_path']]
    def forward(self):
        """run forward pass"""
        self.output=self.netMf(self.left,self.right)
    
    def backward_Mf(self):
        temp1=1-self.critirion1(self.output,self.target)
        temp2=self.critirion2(self.output,self.target)
        self.loss_Mf=temp1+10*temp2
        self.loss_Mf.backward()
        
        
    def optimize_parameters(self):
        self.forward()
        self.optimizer_Mf.zero_grad()
        self.backward_Mf()
        self.optimizer_Mf.step()