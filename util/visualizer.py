
import time
import os
from util import util,html 
import numpy as np


def save_image(webpage,i,visuals,image_path):
    image_dir=webpage.get_image_dir()
    webpage.add_header('test')
    ims, txts, links = [], [], []
    
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%d_%s.png' % ('test',i,label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links)

class Visualizer(object):
    
    def __init__(self,opt):
        self.opt=opt
        self.use_html=util.str2_bool(opt('use_html'))
        if int(opt('display_id'))>0:
            import visdom
            self.vis=visdom.Visdom()
        self.log_path=os.path.join(opt('checkpoints_dir'),opt('name'))
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)    
        
        if self.use_html:
            self.web_dir=os.path.join(opt('checkpoints_dir'),opt('name'),'web')
            self.img_dir=os.path.join(self.web_dir,'images')
            print('create web directory %s \ncreate image result directory %s'%(self.web_dir,self.img_dir))
            util.mkdirs([self.web_dir,self.img_dir])
        
        self.log_name=os.path.join(self.log_path,'loss_log.txt')
        with open(self.log_name,'a') as log_file:
            now=time.strftime('%c')
            log_file.write('===================Training Loss(%s)=================\n' % now)
        


    def display_currnet_results(self,visuals,epoch,save_result):
        """Display current results on visdom.
        
        Parameters:
            visuals (OrderedDict) -- dictionary of images to display or save.
            epoch (int) -- the current epoch.
            save_result (bool) -- if save the results to an HTML file.
        """
        
        if int(self.opt('display_id'))>0:
            ncols=len(visuals)
            h,w=next(iter(visuals.values())).shape[:2]
            table_css = """<style>
                table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                </style>""" % (w, h) 
            title=self.opt('name')
            label_html=''
            label_html_row=''
            images=[]
            idx=0
            for label,image in visuals.items():
                image_numpy=util.tensor2im(image)
                label_html_row+='<td>%s<td>'%label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx+=1
                if idx%ncols==0:
                    label_html+='<tr>%s<tr>'%label_html_row
                    label_html_row=''
            self.vis.images(images, nrow=ncols,padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            # self.vis.text(table_css + label_html,opts=dict(title=title + ' labels'))
        if save_result:
            for label,image in visuals.items():
                image_numpy=util.tensor2im(image)
                image_path=os.path.join(self.img_dir,'epoch%.3f_%s.png'%(epoch,label))
                util.save_image(image_numpy,image_path)
            
            webpage=html.HTML(self.opt,self.web_dir,'Experiment name=%s'%self.opt('name'),refresh=1)
            for n in range(epoch,0,-1):
                webpage.add_header('epoch [%d]'%n)
                ims,txts,links=[],[],[]
                
                for label,image in visuals.items():
                    img_path = 'epoch%.3f_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims,txts,links)
            webpage.save()
            
        


    def print_current_losses(self,epoch,iters,losses,t_comp,t_data):
        """Print current losses on console;and save it to the disk.
        
        Parameters:
            epoch (int) -- current epoch.
            iters (int) -- current training iteration in this epoch.
            losses (OrderedDict) -- current training losses stored in the format of (name,float) pairs. 
            t_comp (float) -- computational time per data point.
            t_data (float) -- data loading time per data point. 
        """
        
        message ='(epoch: %d, iters: %d, time: %.3f, data:%.3f) ' % (epoch,iters,t_comp,t_data)
        for k,v in losses.items():
            message+='loss_%s: %.3f '%(k,v)
        print(message)
        
        with open(self.log_name,'a') as log_file:
            log_file.write('%s\n'%message)
            
    def plot_current_losses(self,epoch,counter_ratio,losses):
        """Display the current losses on visdom display: dictionary of losses labels and values.
        
        Parameter:
            epoch (int) -- currnet epoch.
            counter_ratio (float) -- prograss in currnet epoch.
            losses (OrderedDict) -- training losses stored in the format of (name,float) pairs.
        """
        if not hasattr(self,'plot_data'):
            self.plot_data={'X':[],'Y':[],'legend':list(losses.keys())}
        self.plot_data['X'].append(epoch+counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        
        self.vis.line(X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
                     Y=np.array(self.plot_data['Y']),
                     opts={
                    'title': self.opt('name') + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'})
                