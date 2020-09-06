import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from TPN.models.networks.network_libs import AvgPool3D


class ClsHead(fluid.dygraph.Layer):
    
    def __init__(self, with_avg_pool=True, temporal_feature_size=1,
        spatial_feature_size=7, dropout_ratio=0.8, in_channels=2048,
        num_classes=101, fcn_testing=False, init_std=0.01):
        super(ClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing

        if self.dropout_ratio != 0:
            self.dropout = fluid.layers.dropout
        else:
            self.dropout = None

        if self.with_avg_pool:
            self.avg_pool = AvgPool3D((temporal_feature_size, spatial_feature_size, 
                spatial_feature_size), 1, 0)
        
        if self.fcn_testing:
            self.new_cls = None
            self.in_channels = in_channels
            self.num_classes = num_classes
        
        self.fc_cls = Linear(in_channels, num_classes)
    

    def forward(self, x):
        if not self.fcn_testing:
            if len(x.shape) == 4:
                x = fluid.layers.unsqueeze(x, 2)
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size

            if self.with_avg_pool:
                x = self.avg_pool(x)
            
            if self.dropout is not None:
                x = self.dropout(x, dropout_prob=self.dropout_ratio)
            
            x = fluid.layers.reshape(x, (x.shape[0], -1))
            cls_score = self.fc_cls(x)
            return cls_score
        
        else:
            raise NotImplementedError
    

    def loss(self, cls_score, labels):
        losses = dict()
        loss_cls = fluid.layers.softmax_with_cross_entropy(cls_score, labels)
        losses['loss_cls'] = fluid.layers.reduce_mean(loss_cls)

        return losses