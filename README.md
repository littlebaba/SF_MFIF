# Fully Convolutional network multi-focus image fusion algorithm based on supervised learning 
To improve the quality of multi-focus image fusion, a supervised learning based multi-focus image fusion algorithm based on fully convolutional network is proposed. The aim of this algorithm is to make the neural network learn the complementary relationship between different focusing areas of source images, that is, to select different focusing positions of the source image to synthesize a global clear image. In this algorithm, focusing images are constructed as training data, and dense connection and 1×1 convolution are used in the network to improve the understanding ability and efficiency of the network. The experiment shows that the proposed algorithm is superior to other contrast algorithms in both subjective visual evaluation and objective evaluation, and the quality of image fusion is significantly improved.
<img src="https://github.com/littlebaba/SF_MFIF/blob/master/screenshot/totalframe.png" width='600'>
## Prerequisites
- Python 3.6
- CPU or NVIDIA GPU +CUDA CUDNN
## Getting Started
### Installation
- clone this repo:
~~~bash
git clone git@github.com:littlebaba/SF_MFIF.git
cd SF_MFIF
run train 
~~~
- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
