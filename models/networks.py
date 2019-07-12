import torch
import torch.nn as nn
from torch.nn import init,LeakyReLU,ReLU,ReflectionPad2d,ConstantPad2d,Sigmoid,Tanh
from torch.optim import lr_scheduler
import torch.nn.functional as F
import functools
from math import exp
    
def get_norm_layer(norm_type='batch'):
    """return normalization layer
    
    Paramters:
        norm_type (str) the name of normalization:batch|instance|none
    """
    if norm_type == 'batch':
        norm_layer=functools.partial(nn.BatchNorm2d,affine=True,track_running_stats=True)
    elif norm_type=='instance':
        norm_layer=functools.partial(nn.InstaceNorm2d,affine=False,track_running_stats=False)
    else:
        raise NotImplementedError('normlization layer [%s] is not found.'%norm_type)
    return norm_layer

def init_weight(net,init_type,init_gain):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net,init_type,init_gain,device):
    net.to(device)
    init_weight(net,init_type,init_gain=init_gain)
    return net

def define_Mf(netMf='unet128',norm='batch',use_dropout=False,init_type='normal',init_gain=0.02,device=None):
    """create a mutil focus network.
    
    Paramters:
        netMf (str) -- the architecture's name: unet_128|unet_256|li|vgg_16|vgg_19.
        norm (str) -- the name of normalization layers used in the network:batch|instance:none.
        use_dropout (bool) -- if use dropout layers.
        init_type (str) -- the name of initialization method.
        init_gain (float) -- scaling factor of normal.
        
    return a generator
    
    """
    
    net=None
    norm_layer=get_norm_layer(norm_type=norm)
    
    if netMf=='unet128':
        net=UnetGenerator(1,1,7,norm_layer=norm_layer,use_dropout=use_dropout)
    elif netMf=='p2netV1':
        net=P2netGenerator(1,1,norm_layer=None,use_dropout=False)
    elif netMf=='p2netV2':
        net=P2netGeneratorV2(1,1,norm_layer=None,use_dropout=False)
    elif netMf=='p2netV3':
        net=P2netGeneratorV3(1,1,norm_layer=None,use_dropout=False)
    else:
        raise NotImplementedError('Generator model [%s] is not recognized.'%netMf)
    return init_net(net,init_type,init_gain,device)

def define_D(netD,input_nc=1,ndf=64,n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02,device=None):
    """Create a discriminator.
    
    Parametors:
        input_nc (int)    -- the number of channels in input image.
        ndf (int)         -- the number of filters in the first conv layer.
        netD (int)        -- the type name of discriminator,
        n_layers_D (int)  -- the number of conv layers in the architecture.
        norm (str)        -- the type of normalization layers in the network.
        init_typr (str)   -- the name of initialization method.
        init_gain (int)   -- scaling factors in initialization method.
    """
    net=None
    if netD=='p2DV1':
        net=P2DiscriminatorV1(input_nc,ndf,n_layers_D)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized.'%net)
    return init_net(net,init_type,init_gain,device)
    
    

    
class P2DiscriminatorV1(nn.Module):
    """Define a discriminator."""
    def __init__(self,input_nc,ndf,n_layers,norm_layer=nn.BatchNorm2d):
        """Construct discriminator.
        
        Parametors:
            input_nc (int)        -- the input channels of the input.
            dnf (int)             -- the number of filters in the first layer.
            n_layers(int)       -- the number of conv layer in the network.
            norm_layer (object)   -- the normlization layer.
        """
        super(P2DiscriminatorV1,self).__init__()
        ks=4
        pads=1
        sequence=[nn.Conv2d(input_nc,ndf,kernel_size=ks,stride=2,padding=pads),LeakyReLU(0.2,True)]
        nf_mult=1
        pre_nf_mult=1
        for n in range(1,n_layers):
            pre_nf_mult=nf_mult
            nf_mult=min(2**n,8)
            sequence+=[nn.Conv2d(ndf*pre_nf_mult,ndf*nf_mult,ks,stride=2,padding=pads),
                      norm_layer(ndf*nf_mult),
                      LeakyReLU(0.2,True)]
        pre_nf_mult=nf_mult
        nf_mult=min(2**n_layers,8)
        sequence+=[nn.Conv2d(ndf*pre_nf_mult,ndf*nf_mult,ks,stride=2,padding=pads),
                  norm_layer(ndf*nf_mult),
                  LeakyReLU(0.2,True)]
        sequence+=[nn.Conv2d(ndf*nf_mult,1,ks,stride=2,padding=pads)]
        self.model=nn.Sequential(*sequence)
        
    def forward(self,input):
        return self.model(input)
    
    
    

class P2netGenerator(nn.Module):
    """Create a network for multi-focus images fusion."""
    def __init__(self,input_nc,output_nc,norm_layer=None,use_dropout=False):
        """Construct the generator of this network.
        
        Parameters:
            input_nc (int) -- the number of channels that the network'input.
            output_nc (int) -- the number of channels that the network'output.
            norm_layer (Norm2d) -- the normalization layer.
            use_dropout (bool) -- if use dropout layer.
        """
        super(P2netGenerator, self).__init__()
        self.rdp1oneconv=nn.Conv2d(64,32,1)
        self.rdp1conv=nn.Conv2d(32,96,3)
        self.rdp2oneconv=nn.Conv2d(128,96,1)
        self.rdp2conv=nn.Conv2d(96,160,3)
        self.ddp1oneconv=nn.Conv2d(224,160,1)
        self.ddp1conv=nn.Conv2d(160,96,3)
        self.ddp2oneconv=nn.Conv2d(96,32,1)
        self.ddp2conv=nn.Conv2d(32,output_nc,3)
        
        lkReLU=ReLU(True)
        sigmoid=Sigmoid()
        rpad=ConstantPad2d(1,0)
        
        self.prel=nn.Sequential(rpad,nn.Conv2d(input_nc,16,3),lkReLU,nn.Conv2d(16,8,1),lkReLU,rpad,nn.Conv2d(8,32,3),lkReLU)
        self.prer=nn.Sequential(rpad,nn.Conv2d(input_nc,16,3),lkReLU,nn.Conv2d(16,8,1),lkReLU,rpad,nn.Conv2d(8,32,3),lkReLU)
        
        self.rdp1=nn.Sequential(self.rdp1oneconv,lkReLU,rpad,self.rdp1conv,lkReLU)
        self.rdp2=nn.Sequential(self.rdp2oneconv,lkReLU,rpad,self.rdp2conv,lkReLU)
        
        self.dd=nn.Sequential(self.ddp1oneconv,lkReLU,rpad,self.ddp1conv,lkReLU,self.ddp2oneconv,lkReLU,rpad,self.ddp2conv,Tanh())
    def forward(self,left,right):
        l=self.prel(left)
        r=self.prer(right)
        cat1=torch.cat([l,r],1)
        cat2=torch.cat([self.prel[:2](left),self.rdp1(cat1),self.prer[:2](right)],1)
        cat3=torch.cat([l,self.rdp2(cat2),r],1)
        return self.dd(cat3)
        

class P2netGeneratorV2(nn.Module):
    """V2: Create a network for multi-focus images fusion."""
    def __init__(self,input_nc,output_nc,norm_layer=None,use_dropout=False):
        """Construct the generator of this network.
        
        Parameters:
            input_nc (int) -- the number of channels that the network'input.
            output_nc (int) -- the number of channels that the network'output.
            norm_layer (Norm2d) -- the normalization layer.
            use_dropout (bool) -- if use dropout layer.
        """
        super(P2netGeneratorV2, self).__init__()
              
        lkReLU=Tanh()
        rpad=ConstantPad2d(1,0)
        
        self.prel=nn.Sequential(rpad,nn.Conv2d(input_nc,16,3),lkReLU,nn.Conv2d(16,8,1),lkReLU,rpad,nn.Conv2d(8,32,3),lkReLU,nn.Conv2d(32,16,1),lkReLU,rpad,nn.Conv2d(16,64,3),lkReLU)
        self.prer=nn.Sequential(rpad,nn.Conv2d(input_nc,16,3),lkReLU,nn.Conv2d(16,8,1),lkReLU,rpad,nn.Conv2d(8,32,3),lkReLU,nn.Conv2d(32,16,1),lkReLU,rpad,nn.Conv2d(16,64,3),lkReLU)
        
        self.rdp1=nn.Sequential(nn.Conv2d(128,64,1),lkReLU,nn.Conv2d(64,64,1),lkReLU,rpad,nn.Conv2d(64,192,3),lkReLU)
        self.rdp2=nn.Sequential(nn.Conv2d(224,128,1),lkReLU,rpad,nn.Conv2d(128,256,3),lkReLU)
        self.rdp3=nn.Sequential(nn.Conv2d(320,224,1),lkReLU,rpad,nn.Conv2d(224,384,3),lkReLU)
        
        self.dd=nn.Sequential(nn.Conv2d(512,128,1),lkReLU,nn.Conv2d(128,128,1),lkReLU,rpad,nn.Conv2d(128,64,3),lkReLU,rpad,nn.Conv2d(64,32,3),lkReLU,rpad,nn.Conv2d(32,1,3),lkReLU)
    def forward(self,left,right):
        l=self.prel(left)
        r=self.prer(right)
        temp1=torch.cat([l,r],1)#128
        cat1=self.rdp1(temp1)#192
        
        temp2=torch.cat([self.prel[:2](left),cat1,self.prer[:2](right)],1)#224
        cat2=self.rdp2(temp2)#256
        
        temp3=torch.cat([self.prel[:8](left),cat2,self.prer[:8](right)],1)#320
        cat3=self.rdp3(temp3)#384
        
        temp4=torch.cat([l,cat3,r],1)#512
        return self.dd(temp4)
        
class P2netGeneratorV3(nn.Module):
    """not use 1x1 conv,use residual block structure;every layer use ReLU activation except last one,and tanh is used for last layer.
    
    General Structure: === l/r->conv(1,64,9)->resblock1->resblock2->cat->resblock3->resblock4->conv(64,64,3)->conv(64,64,3)-> conv(64,64,3)-> conv(64,1,3)-> output ===
    Residual Block:    === feature_in -> conv(64,64,3)->conv(64,64,3)+feature_in
    """
    def __init__(self,input_nc,output_nc,norm_layer=None,use_dropout=False):
        """Initialize this network.
        
        Parameters:
            input_nc (int) -- the number of channels that the network'input.
            output_nc (int) -- the number of channels that the network'output.
            norm_layer (object) -- the normalization layer.
            use_drop (bool) -- if use dropout layer.
        """
        super(P2netGeneratorV3,self).__init__()
        self.input_nc=input_nc
        self.output_nc=output_nc
        self.activate=ReLU(True)
        self.pad=ConstantPad2d(1,0)
        self.fea_in=nn.Sequential(ConstantPad2d(4,0),nn.Conv2d(self.input_nc,64,9),self.activate)
        self.rb1=nn.Sequential(self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate)
        self.rb2=nn.Sequential(self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate)
        self.cat=nn.Sequential(self.pad,nn.Conv2d(128,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate)
        self.rb3=nn.Sequential(self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate)
        self.rb4=nn.Sequential(self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate)
        self.after=nn.Sequential(self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,64,3),self.activate,self.pad,nn.Conv2d(64,1,3),Tanh())

    def twoblock(self,input):
        temp=self.fea_in(input)
        temp1=self.rb1(temp)
        temp2=self.rb2(temp1)
        return temp2
    def forward(self,left,right):
        l=self.twoblock(left)
        r=self.twoblock(right)
        cat=torch.cat([l,r],1)
        temp=self.cat(cat)
        p1=self.rb3(temp)
        p2=self.rb4(p1)
        return self.after(p2)
        
class MfLoss():
    
    pass



def gaussian(window_size,sigma=1.5):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
    
def create_window(window_size,channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
    
def _ssim(output, target, window, window_size, channel,size_average):
    mu1 = F.conv2d(target, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(output, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(target*target, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(output*output, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(target*output, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    """Define SSIM loss function."""
    
    def __init__(self,window_size = 11, size_average = True):
        """Initialize ssim
        
        Parameters:
            window_size (int) -- the size of slide window.
            size_average (bool) -- if the result of ssim funtion need to average. 
        """
        super(SSIMLoss,self).__init__()
        self.window_size=window_size
        self.size_average=size_average
        self.channel=1
        self.window=create_window(window_size,self.channel)
          
    def forward(self,output,target):
        (_, channel, _, _) = target.size()
        if channel == self.channel and self.window.data.type() == target.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
#            if target.is_cuda:
#                window = window.cuda(img1.get_device())
            window = window.type_as(target)
            self.window = window
            self.channel = channel
        return _ssim(output, target, window, self.window_size,channel,self.size_average)
        

    
    
class UnetGenerator(nn.Module):
    """create a unet-based generator.
    
    """
    def __init__(self,output_nc,input_nc,num_downs,norm_layer,use_dropout=False,ngf=64):
        """Construct a unet generator
        
        paramters:
            num_downs (int) -- the number of downsamplings in unet.
        
        We construct the unet from the innermost layer to the outermost layer.
        """
        super(UnetGenerator,self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer) # add the outermost layer
        
    def forward(self,input):
        return self.model(input)

  
        
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        |--downsampling---submodule---upsampling---|
    """
    def __init__(self,outer_nc,inner_nc,input_nc=None,submodule=None,innermost=False,outermost=False,norm_layer=nn.BatchNorm2d,use_dropout=False):
        """Construct a Unet submodule with skip connection.
            
        Paramters:
            
        """
        super(UnetSkipConnectionBlock,self).__init__()
        self.outermost=outermost
        if type(norm_layer)==functools.partial:
            use_bias=norm_layer.func==nn.InstanceNorm2d
        else:
            use_bias=norm_layer==nn.InstanceNorm2d
        if input_nc==None:
            input_nc=outer_nc
        downconv=nn.Conv2d(input_nc,inner_nc,kernel_size=4,stride=2,padding=1,bias=use_bias)
        downrelu=nn.LeakyReLU(0.2,True)
        downnorm=norm_layer(inner_nc)
        uprelu=nn.ReLU(True)
        upnorm=norm_layer(outer_nc)
        
        if outermost:
            upconv=nn.ConvTranspose2d(inner_nc*2,outer_nc,kernel_size=4,stride=2,padding=1)
            down=[downconv]
            up=[uprelu,upconv,upnorm]
            model=down+[submodule]+up
        elif innermost:
            upconv=nn.ConvTranspose2d(inner_nc,outer_nc,kernel_size=4,stride=2,padding=1,bias=use_bias)
            down=[downrelu,downconv]
            up=[uprelu,upconv,upnorm]
            model=down+up
        else:
            upconv=nn.ConvTranspose2d(inner_nc*2,outer_nc,kernel_size=4,stride=2,padding=1,bias=use_bias)
            down=[downrelu,downconv,downnorm]
            up=[uprelu,upconv,upnorm]
            if use_dropout:
                model=down+[submodule]+up+nn.Dropout(0.5)
            else:
                model=down+[submodule]+up
                
        self.model=nn.Sequential(*model)
    
    def forward(self,x):
        if self.outermost:
            return self.model(x)
        else: #add skip connection.
            return torch.cat([x,self.model(x)],1)

class GANLoss(nn.Module):
    """The goal of this class is to create the target label tensor that has the same size as the input."""
    def __init__(self,gan_mode,target_real_label=1.0,target_fake_label=0.0):
        """Initialize the GANLoss class.
        
        Parametors:
            gan_mode (str)    -- the type of GAN objective.It currently supports vanilla,lsgan,and wgangp.
            target_real_label -- label for real image.
            target_fake_label -- label for fake image.
        """
        super(GANLoss,self).__init__()
        self.register_buffer('real_label',torch.tensor(target_real_label))
        self.register_buffer('fake_label',torch.tensor(target_fake_label))
        self.gan_mode=gan_mode
        if gan_mode=='lsgan':
            self.loss=nn.MSELoss()
        elif gan_mode=='vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss=None
        else:
            raise NotImplementedError('gan mode %s not implemented'%gan_mode)
    
    def get_target_tensor(self,prediction,target_is_real):
        """Create label tensors with the same size as the_input.
        
        Parametors:
            prediction (tensor)   -- typically the prediction from a discriminator.
            target_is_real (bool) -- 
        Returns:
            a label tensor filled with ground truth albel,and with the size of input.
        """
        
        if target_is_real:
            target_tensor=self.real_label
        else:
            target_tensor=self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self,prediction,target_is_real):
        """give descriminator's output and ground truth label and calculate loss.
        
        Parameters:
            prediction (tensor)    -- typically the prediction output from a deciminator.
            target_is_real (bool)  -- 
        Returns:
            the calculated loss
        """
        
        if self.gan_mode in ['lsgan','vanilla']:
            target_tensor = self.get_target_tensor(prediction,target_is_real)
            loss=self.loss(prediction,target_tensor)
        
        return loss