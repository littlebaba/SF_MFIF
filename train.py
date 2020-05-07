# %%

import functools
import time
from configparser import ConfigParser

from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torchvision import transforms


def train():
    cp = ConfigParser()
    cp.read('conf.cfg', encoding="utf-8")
    # cp.read('conf_GAN.cfg', encoding='utf-8')
    opt = functools.partial(cp.get, cp.sections()[0])

    dataset = create_dataset(opt)
    print('The number of training images = %d' % len(dataset))
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(1, 100 + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):

            iter_start_time = time.time()
            if total_iters % int(opt('print_freq')) == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += int(opt('batch_size'))
            epoch_iter += int(opt('batch_size'))
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % int(opt('print_freq')) == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / int(opt('batch_size'))
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, epoch_iter / len(dataset), losses)

            if total_iters % int(opt('display_freq')) == 0:
                visualizer.display_currnet_results(model.get_currnet_visuals(), epoch, save_result=True)
            if total_iters % int(opt('save_latest_freq')) == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters
                model.save_networks(save_suffix)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, 100, time.time() - epoch_start_time))


if __name__ == '__main__':
    train()
