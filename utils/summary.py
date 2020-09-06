import numpy as np 
import os
import sys
sys.path.append('..')
from utils.config import cfg
from tensorboardX import SummaryWriter


def mkdirs(newdir):
    
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


class AverageMeter(object):

	def __init__(self):
		self.reset()


	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0


	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class LogSummary(object):

    def __init__(self, log_path):
        mkdirs(log_path)
        self.writer = SummaryWriter(log_path)
    

    def write_scalars(self, scalar_dict, n_iter, tag=None):
        for name, scalar in scalar_dict.items():
            if tag is not None:
                name = '/'.join([tag, name])
            self.writer.add_scalar(name, scalar, n_iter)
    

    def write_hist_parameters(self, net, n_iter):
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().numpy(), n_iter)
