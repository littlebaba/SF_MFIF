import torch
from sewar.full_ref import mse as mse_
from sewar.full_ref import psnr as psrn_
from sewar.full_ref import scc as scc_
from sewar.full_ref import ssim as ssim_
from sewar.full_ref import uqi as uqi_
from sewar.full_ref import vifp as vifp_
from torch import Tensor

lam = lambda x: x.data.squeeze(0).permute((1, 2, 0)).numpy()


def mse(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return mse_(lam(s), lam(t))


def psrn(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return psrn_(lam(s), lam(t), MAX=1)


def uqi(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return uqi_(lam(s), lam(t))


def ssim(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return ssim_(lam(s), lam(t), MAX=1)[0]


def scc(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return scc_(lam(s), lam(t))


def vifp(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return vifp_(lam(s), lam(t))


def metric(s: Tensor, t: Tensor):
    """
    6 metrics: mse, psrn, qui, ssim, scc, vifp
    :param s:
    :param t:
    :return:
    """
    return {'MSE': mse(s, t), 'PSRN': psrn(s, t), 'QUI': uqi(s, t),
            'SSIM': ssim(s, t), 'SCC': scc(s, t), 'VIFP': vifp(s, t)}


if __name__ == '__main__':
    s = torch.randn((1, 3, 88, 88))
    t = torch.randn((1, 3, 88, 88))
    ret = metric(s, t)
    print(ret)
