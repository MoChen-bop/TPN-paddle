import os
import sys
sys.path.append('/home/aistudio')
import math
import numpy as np
from datetime import datetime

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

from TPN.models.tsn3d import TPN3D
from TPN.readers.rawframes_dataset import RawFramesDataset
from TPN.utils.config import cfg
from TPN.utils.summary import AverageMeter, LogSummary
from TPN.utils.timer import Timer


class Solver():

    def __init__(self, data_loader):
        self.data_loader = data_loader

        self.model = TPN3D(cfg.models)
        self.model.train()

        self.optimizer = self.get_optimizer(self.model.parameters())

        log_dir = os.path.join(cfg.solver.log_dir, cfg.exp_name, 'train', datetime.now().strftime('%b%d_%H-%M-%S_'))
        self.logger = LogSummary(log_dir)

        self.global_loss = AverageMeter()
        self.global_loss_cls = AverageMeter()
        self.global_loss_aux = AverageMeter()
        self.global_cls_acc1 = AverageMeter()
        self.global_cls_acc5 = AverageMeter()
        self.global_aux_acc1 = AverageMeter()
        self.global_aux_acc5 = AverageMeter()
        self.global_mix_acc1 = AverageMeter()
        self.global_mix_acc5 = AverageMeter()

        self.timer = Timer()
        self.timer.start()
    

    def read_data(self, ):
        for step, data in enumerate(self.data_loader()):
            img_group = np.array([x[0] for x in data]).astype('float32')
            labels = np.array([x[1] for x in data]).astype('int64')

            img_group = to_variable(img_group)
            labels = to_variable(labels)

            yield step, (img_group, labels)
    

    def forward(self, img_group, labels, for_test=False):
        loss, accuracy = self.model(img_group, labels, for_test)
        return loss, accuracy
    

    def backward(self, loss):
        loss_sum = loss['loss_cls'] + loss['loss_aux']
        loss_sum.backward()
        self.optimizer.minimize(loss_sum)
        self.model.clear_gradients()


    def update_logger(self, epoch, step, loss, accuracy, save=True):
        global_step = epoch * cfg.data.hmdb.train_len // cfg.solver.batch_size + step
        loss_cls = loss['loss_cls'].numpy()[0]
        loss_aux = loss['loss_aux'].numpy()[0]
        loss = loss_cls + loss_aux
        cls_acc1 = accuracy['cls_acc1'].numpy()[0]
        cls_acc5 = accuracy['cls_acc5'].numpy()[0]
        aux_acc1 = accuracy['aux_acc1'].numpy()[0]
        aux_acc5 = accuracy['aux_acc5'].numpy()[0]
        mix_acc1 = accuracy['mix_acc1'].numpy()[0]
        mix_acc5 = accuracy['mix_acc5'].numpy()[0]

        self.global_loss.update(loss)
        self.global_loss_cls.update(loss_cls)
        self.global_loss_aux.update(loss_aux)
        self.global_cls_acc1.update(cls_acc1)
        self.global_cls_acc5.update(cls_acc5)
        self.global_aux_acc1.update(aux_acc1)
        self.global_aux_acc5.update(aux_acc5)
        self.global_mix_acc1.update(mix_acc1)
        self.global_mix_acc5.update(mix_acc5)

        if step % cfg.solver.log_interval == 0:
            speed = cfg.solver.log_interval / self.timer.elapsed_time()
            print(("Epoch: {}, Step: {}, Loss: {:.2f}[{:.2f}], Loss_cls: {:.2f}[{:.2f}],"
                   " Loss_aux: {:.2f}[{:.2f}], Acc_cls@1: {:.2f}[{:.2f}], Acc_cls@5: {:.2f}[{:.2f}],"
                   " Acc_aux@1: {:.2f}[{:.2f}], Acc_aux@5: {:.2f}[{:.2f}], Acc_mix@1: {:.2f}[{:.2f}], Acc_mix@5: {:.2f}[{:.2f}],"
                   " LR: {:.5f}, Speed: {:.1f} step / second").format(epoch, global_step, self.global_loss.avg, loss, 
                        self.global_loss_cls.avg, loss_cls, self.global_loss_aux.avg, loss_aux, 
                        self.global_cls_acc1.avg, cls_acc1, self.global_cls_acc5.avg, cls_acc5,
                        self.global_aux_acc1.avg, aux_acc1, self.global_aux_acc5.avg, aux_acc5,
                        self.global_mix_acc1.avg, mix_acc1, self.global_mix_acc5.avg, mix_acc5,
                        self.optimizer.current_step_lr(), speed))
            
            if save:
                self.logger.write_scalars({
                    'global_loss': self.global_loss.avg,
                    'global_loss_cls': self.global_loss_cls.avg,
                    'global_loss_aux': self.global_loss_aux.avg,
                    'loss': loss,
                    'loss_cls': loss_cls,
                    'loss_aux': loss_aux,
                    'global_cls_acc1': self.global_cls_acc1.avg,
                    'global_cls_acc5': self.global_cls_acc5.avg,
                    'global_aux_acc1': self.global_aux_acc1.avg,
                    'global_aux_acc5': self.global_aux_acc5.avg,
                    'global_mix_acc1': self.global_mix_acc1.avg,
                    'global_mix_acc5': self.global_mix_acc5.avg,
                    'cls_acc1': cls_acc1,
                    'cls_acc5': cls_acc5,
                    'aux_acc1': aux_acc1,
                    'aux_acc5': aux_acc5,
                    'mix_acc1': mix_acc1,
                    'mix_acc5': mix_acc5,
                }, tag='train', n_iter=global_step)

            self.timer.restart()


    def get_optimizer(self, parameter_list):
        step = int(math.ceil(float(cfg.data.hmdb.train_len) / cfg.solver.batch_size))
        epochs = cfg.solver.lr_decay_epoch
        bd = [step * e for e in epochs]
        lr = []
        lr = [cfg.solver.base_lr * (0.1 ** i) for i in range(len(bd) + 1)]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(boundaries=bd, values=lr),
            momentum=cfg.solver.momentum_rate,
            regularization=fluid.regularizer.L2Decay(cfg.solver.l2_decay),
            parameter_list=parameter_list)
        return optimizer
    

    def save_models(self, epoch):
        save_path = os.path.join(cfg.solver.save_dir, cfg.exp_name, str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        fluid.save_dygraph(self.model.state_dict(), os.path.join(save_path, 'TPN.params'))
    

    def load_models(self, epoch):
        save_path = os.path.join(cfg.solver.save_dir, cfg.exp_name, str(epoch), 'TPN.params.pdparams')
        if not os.path.exists(save_path):
            print('cannot find path: ' + save_path)
            exit(1)
        print('Loading pretrained from path: ' + save_path)

        param_dict, _ = fluid.dygraph.load_dygraph(save_path)
        self.model.load_dict(param_dict)