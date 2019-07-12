from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self,parser):
        BaseOptions.initialize(self,parser)
        parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        return parser