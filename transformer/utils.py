import torch
from torch.autograd import Variable
import numpy as np


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np.mask.cuda()
    return np_mask


def create_mask(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


