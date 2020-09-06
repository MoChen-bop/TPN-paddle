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


class Evaluator():

    def __init__(self, data_loader,):
        self.data_loader = data_loader
        
        self.model = TPN3D(cfg.models)
        self.model.eval()

        log_dir = os.path.join(cfg.solver.log_dir, cfg.exp_name, 'eval', datetime.now().strftime('%b%d_%H-%M-%S_'))
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
        
        self.loss_cls = AverageMeter()
        self.loss_aux = AverageMeter()
        self.cls_acc1 = AverageMeter()
        self.cls_acc5 = AverageMeter()
        self.aux_acc1 = AverageMeter()
        self.aux_acc5 = AverageMeter()
        self.mix_acc1 = AverageMeter()
        self.mix_acc5 = AverageMeter()

        self.timer = Timer()
        self.timer.start()
    

    def read_data(self, ):
        for step, data in enumerate(self.data_loader()):
            img_group = np.array([x[0] for x in data]).astype('float32')
            labels = np.array([x[1] for x in data]).astype('int64')

            img_group = to_variable(img_group)
            labels = to_variable(labels)

            yield step, (img_group, labels)
    

    def forward(self, img_group, labels, for_test=True):
        loss, accuracy = self.model(img_group, labels, for_test)
        return loss, accuracy


    def reset_summary(self):
        self.loss_cls.reset()
        self.loss_aux.reset()
        self.cls_acc1.reset()
        self.cls_acc5.reset()
        self.aux_acc1.reset()
        self.aux_acc5.reset()
        self.mix_acc1.reset()
        self.mix_acc5.reset()


    def update_logger(self, epoch, step, loss, accuracy, save=True):
        global_step = epoch * cfg.data.hmdb.val_len // cfg.eval.solver.batch_size + step
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

        self.loss_cls.update(loss_cls)
        self.loss_aux.update(loss_aux)
        self.cls_acc1.update(cls_acc1)
        self.cls_acc5.update(cls_acc5)
        self.aux_acc1.update(aux_acc1)
        self.aux_acc5.update(aux_acc5)
        self.mix_acc1.update(mix_acc1)
        self.mix_acc5.update(mix_acc5)

        if step % cfg.solver.log_interval == 0:
            speed = cfg.solver.log_interval / self.timer.elapsed_time()
            print(("Epoch: {}[{}], L: {:.1f}[{:.1f}], L_cls: {:.1f}[{:.1f}],"
                   " L_aux: {:.1f}[{:.1f}], A_cls@1: {:.1f}[{:.1f}], A_cls@5: {:.1f}[{:.1f}],"
                   " A_aux@1: {:.1f}[{:.1f}], A_aux@5: {:.1f}[{:.1f}], A_mix@1: {:.1f}[{:.1f}], A_mix@5: {:.1f}[{:.1f}],"
                   " Speed: {:.1f} step / second").format(epoch, global_step, self.global_loss.avg, loss, 
                        self.global_loss_cls.avg, loss_cls, self.global_loss_aux.avg, loss_aux, 
                        self.global_cls_acc1.avg, cls_acc1, self.global_cls_acc5.avg, cls_acc5,
                        self.global_aux_acc1.avg, aux_acc1, self.global_aux_acc5.avg, aux_acc5,
                        self.global_mix_acc1.avg, mix_acc1, self.global_mix_acc5.avg, mix_acc5,
                        speed))
            
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
                }, tag='eval', n_iter=global_step)

            self.timer.restart()


    def write_summary(self, epoch):
        save_result_dir = os.path.join(cfg.solver.log_dir, cfg.exp_name, 'eval_result')
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
        with open(os.path.join(save_result_dir, str(epoch) + '.txt'), 'w') as f:
            f.write("cls_acc1: {:.3f}, cls_acc5: {:.3f}, aux_acc1: {:.3f}, aux_acc5: {:.3f}, mix_acc1: {:.3f}, mix_acc5: {:.3f}\n".format(
                self.cls_acc1.avg, self.cls_acc5.avg, self.aux_acc1.avg, self.aux_acc5.avg, self.mix_acc1.avg, self.mix_acc5.avg
            ))


    def load_models(self, epoch):
        save_path = os.path.join(cfg.solver.save_dir, cfg.exp_name, str(epoch), 'TPN.params.pdparams')
        if not os.path.exists(save_path):
            print('cannot find path: ' + save_path)
            exit(1)
        print('Loading pretrained from path: ' + save_path)

        param_dict, _ = fluid.dygraph.load_dygraph(save_path)
        self.model.load_dict(param_dict)