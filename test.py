import os
import time
from configparser import ConfigParser
from functools import partial

import numpy as np

import pytorch_metric
from data import create_dataset
from models import create_model
from util import html, visualizer


def test():
    cp = ConfigParser()
    cp.read('conf.cfg', encoding='utf-8')
    # cp.read('conf_GAN.cfg',encoding='utf-8')
    opt = partial(cp.get, cp.sections()[1])
    dataset = create_dataset(opt)
    print('the number of testing images=%d' % len(dataset))
    model = create_model(opt)
    model.setup(opt)
    web_dir = os.path.join(opt('checkpoints_dir'), opt('name'), opt('results_dir'))
    webpage = html.HTML(opt, web_dir, 'Experiment = %s, Phase = %s' % (opt('name'), opt('phase')))
    model.eval()
    with open(os.path.join(webpage.get_image_dir(), 'record.txt'), 'w') as file:
        file.write(str(cp.items('train')))
    s = time.time()
    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()
        mse, psrn, uqi, ssim, scc, vifp = np.array(list(pytorch_metric.metric(model.left, model.output).values())) * 0.5 + \
                                          np.array(list(pytorch_metric.metric(model.right, model.output).values())) * 0.5
        test_msg = f'MSE: {mse}, PSRN: {psrn}, UQI: {uqi}, SSIM: {ssim}, SCC: {scc}, VIFP: {vifp}'
        print(test_msg)
        visuals = model.get_currnet_visuals()
        img_path = model.get_image_paths()
        print('processing (%04d)-th image... %s' % (i, img_path))
        visualizer.save_image(webpage, i, visuals, img_path)
    e = time.time()
    print('Time consuming is %f' % ((e - s) / 27))
    webpage.save()


if __name__ == '__main__':
    test()
