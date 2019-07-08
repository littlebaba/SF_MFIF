# 基于监督学习的全卷积神经网络多聚焦融合算法
使用pytorch实现的一种基于监督学习的全卷积神经网络多聚焦图像融合算法。
该算法旨在使神经网络学习到一个源图像不同聚焦区域的互补关系，
即选择出源图像中不同的聚焦位置合成一张全局清晰图像。
为提高网络的理解能力和效率，该算法构造聚焦图像作为训练数据，
并且网络采用了稠密连接和1x1卷积核。
<img src="https://github.com/littlebaba/SF_MFIF/blob/master/screenshot/totalframe.png" width='600'>

hh