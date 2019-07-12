import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os
from util.util import *
class HTML(object):
    """This class to allow us to save images and write texts into s single html file.
    
    It consists of the following functions:
        <add_header> -- add a text header to the html file.
        <add_images> -- add a row of images to the html file.
        <save> -- save the html to disk.
        It is based on Python library 'dominate'.
     """
    
    def __init__(self,opt,web_dir,title,refresh=0):
        """Initialize the HTML file.
        
        Parameters:
            web_dir (str) -- a directory that stores the webpage.
            title (str) -- the webpage's name
            refresh (int) -- how ofthe the webpage refresh itself.If 0,no refreshing.
        """
        self.opt=opt
        self.web_dir=web_dir
        self.title=title
        self.image_dir = os.path.join(self.web_dir,'images')
        if not os.path.exists(self.web_dir):
            os.mkdir(self.web_dir)
        
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)
            
        self.doc=dominate.document(title=self.title)
        if refresh>0:
            with self.doc.head:
                meta(http_equive='refresh',content=str(refresh))
    
    def add_header(self,text):
        """Insert a header to the html file.
        
        Parameters:
            text (str) -- the header of html.
        """
        with self.doc:
            h3(text)
            
    def get_image_dir(self):
        return self.image_dir
    
    def add_images(self,ims,txts,links,width=400):
        """add images to the HTML file.
        
        Parameters:
            ims (str list) -- a list of image path.
            txts (str list) -- a list of image names shown in webpage.
            links (str list) -- a list of hyperref links;when you click an image,it will redirect you to a new page. 
        """
        self.t=table(border=1,style="table-layout: fixed;")
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im,txt,link in zip(ims,txts,links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)
        
    
    def save(self):
        """Save the current content to the html file."""
        html_file='%s/index.html'%self.web_dir
        with open(html_file,'wt') as file:
            file.write(self.doc.render())

        