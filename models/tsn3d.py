import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from TPN.models.networks.backbone.resnet_3d import ResNet3D
from TPN.models.networks.backbone.xception_3d import Xception
from TPN.models.networks.tpn import TPN
from TPN.models.networks.simple_spatial_temporal_module import SimpleSpatialTemporalModule
from TPN.models.networks.simple_consensus import SimpleConsenus
from TPN.models.networks.cls_head import ClsHead


class TPN3D(fluid.dygraph.Layer):

    def __init__(self, model_cfg):
        super(TPN3D, self).__init__()
        if model_cfg.backbone_name == 'resnet':
            self.backbone = ResNet3D(**model_cfg.backbone)
        else:
            self.backbone = Xception(**model_cfg.backbone)
        self.necks = TPN(**model_cfg.tpn)
        self.spatial_temporal_module = SimpleSpatialTemporalModule()
        self.segmental_consensus = SimpleConsenus()
        self.cls_head = ClsHead(**model_cfg.cls_head)

    
    def forward(self, img_group, gt_label, for_test=False):
        b, n, c, t, h, w = img_group.shape
        x = fluid.layers.reshape(img_group, (-1, c, t, h, w))
        label = fluid.layers.reshape(gt_label, (b * n, -1))

        x = self.backbone(x)
        x, aux_losses, aux_accuracy, aux_score = self.necks(x, label, for_test)
        x = self.spatial_temporal_module(x)
        x = fluid.layers.reshape(x, (b, n, x.shape[1], x.shape[2], x.shape[3], x.shape[4])) # (b, n, c, t, 1, 1)
        x = self.segmental_consensus(x) # (b, 1, c, t, 1, 1)
        x = fluid.layers.squeeze(x, [1]) # (b, c, t, 1, 1)
        losses, accuracy = dict(), dict()
        cls_score = self.cls_head(x)
        loss_cls = self.cls_head.loss(cls_score, gt_label[:, 0])
        losses.update(loss_cls)
        losses.update(aux_losses)
        accuracy.update(aux_accuracy)
        
        cls_score = fluid.layers.softmax(cls_score, 1)
        accuracy['cls_acc1'] = fluid.layers.accuracy(input=cls_score, label=gt_label[:,0], k=1)
        accuracy['cls_acc5'] = fluid.layers.accuracy(input=cls_score, label=gt_label[:,0], k=5)
        
        mix_score = (cls_score + aux_score) / 2
        accuracy['mix_acc1'] = fluid.layers.accuracy(input=mix_score, label=gt_label[:,0], k=1)
        accuracy['mix_acc5'] = fluid.layers.accuracy(input=mix_score, label=gt_label[:,0], k=5)

        return losses, accuracy


if __name__ == '__main__':
    import sys
    sys.path.append('/home/aistudio/')
    import numpy as np
    from paddle.fluid.dygraph import to_variable
    from TPN.utils.config import cfg

    with fluid.dygraph.guard():
        img_group = to_variable(np.zeros((1, 1, 3, 32, 224, 224)).astype('float32'))
        gt_label = to_variable(np.zeros((1, 1, 1)).astype('int64'))

        model = TPN3D(cfg.models)

        losses, _ = model(img_group, gt_label)

        print(losses['loss_cls'].shape)
        print(losses['loss_aux'].shape)

    

    