import argparse




class BaseOptions(object):
    
    def __init__(self):
        self.initialized=False
    
    def initialize(self,parser):
        parser.add_argument('--batch_size',type=int,default=25,help='input batch size')
        self.initialized=True
        return parser
    
    def gather_options(self):
        if not self.initialized:
            parser=argparse.ArgumentParser()
            parser=self.initialize(parser)
        self.parser=parser
        return parser.parse_args(args=[])
    
    def print_options(self,opt):
        message=''
        message+='-----------------options--------------\n'
        for k,v in sorted(vars(opt).items()):
            message+='{:>20}: {:<25}\n'.format(str(k),v)
        message+='------------------end------------------'
        print(message)
    def parse(self):
        opt=self.gather_options()
        self.print_options(opt)
        return opt
    