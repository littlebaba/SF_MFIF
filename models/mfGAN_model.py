import torch
from models.base_model import BaseModel
from . import networks

class MfGANModel(BaseModel):
    """The class is to create GAN that applied to deal with multi-focus image fusion."""
    
    def __init__(self,opt):
        """Initialize the GAN."""
        BaseModel.__init__(self,opt)
        self.loss_names=['G','D'];
        if self.isTrain:
            self.model_names=['G','D']
            self.visual_names=['left','right','target','fake'];
        else:
            self.model_names=['G']
            self.visual_names=['left','right','fake'];
        self.netG=networks.define_Mf(opt('netG'),device=self.device)
        if self.isTrain:
            self.netD=networks.define_D(opt('netD'),device=self.device)
        if self.isTrain:
            self.criterionGAN=networks.GANLoss('vanilla').to(self.device)
            self.criterionMSE=networks.GANLoss('lsgan').to(self.device)
            self.optimizer_G=torch.optim.Adam(self.netG.parameters(),lr=0.0005)
            self.optimizer_D=torch.optim.Adam(self.netD.parameters(),lr=0.0001)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def set_input(self,input):
        
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
        self.fake=self.netG(self.left,self.right)
        
    def backward_D(self):
        """Calculate the loss of discriminator."""
        temp_fake=self.netD(self.fake.detach())
        self.loss_D_fake=self.criterionGAN(temp_fake,False)
        temp_real=self.netD(self.target)
        self.loss_D_real=self.criterionGAN(temp_real,True)
        self.loss_D=(self.loss_D_fake+self.loss_D_real)*0.5
        self.loss_D.backward()
    
    def backward_G(self):
        """Calculate the loss of generator."""
        fake=self.netD(self.fake)
        self.loss_G=self.criterionMSE(fake,True)
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()
        #update netD
        self.set_requires_grad(self.netD,True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        #update netG
        self.set_requires_grad(self.netD,False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        