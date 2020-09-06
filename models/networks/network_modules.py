import sys
sys.path.append('/home/aistudio')
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear

from TPN.models.networks.network_libs import Conv1x3x3, Conv3x1x1
from TPN.models.networks.network_libs import MaxPool3D, Conv3DBN


class Identity(fluid.dygraph.Layer):

    def __init__(self, ):
        super(Identity, self).__init__()
    

    def forward(self, x):
        return x


class Upsampling(fluid.dygraph.Layer):

    def __init__(self, scales=(2, 1, 1)):
        super(Upsampling, self).__init__()
        assert scales[1] == 1
        assert scales[2] == 1
        self.scale = scales

    def forward(self, inputs):
        b, c, t, h, w = inputs.shape
        scale = (t * self.scale[0], h * self.scale[1] * w * self.scale[2])
        inputs = fluid.layers.reshape(inputs, (b, c, t, h * w))
        out = fluid.layers.interpolate(inputs, scale, resample='NEAREST')
        out = fluid.layers.reshape(out, (b, c, scale[0], h, w))
        return out


class Downsampling(fluid.dygraph.Layer):

    def __init__(self, inplanes, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1),
        downsample_position='after', downsample_scale=(1, 2, 2)):
        super(Downsampling, self).__init__()

        self.conv = Conv3DBN(inplanes, planes, kernel_size, stride, act='relu')
        self.pool = MaxPool3D(downsample_scale, downsample_scale)
        self.downsample_position = downsample_position

    def forward(self, inputs):
        if self.downsample_position == 'before':
            inputs = self.pool(inputs)

        x = self.conv(inputs)
        x = self.pool(x)

        if self.downsample_position == 'after':
            x = self.pool(x)
    
        return x


class TemporalModulation(fluid.dygraph.Layer):

    def __init__(self, inplanes, planes, downsample_scale=8):
        super(TemporalModulation, self).__init__()

        self.conv = Conv3x1x1(inplanes, planes)
        self.pool = MaxPool3D((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0))
    

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x


class SpatialModulation(fluid.dygraph.Layer):

    def __init__(self, inplanes=[1024, 2048], planes=2048):
        super(SpatialModulation, self).__init__()

        self.spatial_modulation = []
        for i, dim in enumerate(inplanes):
            ops = []
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                ops = self.add_sublayer('spatial_up_%d_0' % i, Identity())
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    up = self.add_sublayer('spatial_up_%d_%d' % (i, dsi),
                        Conv1x3x3(dim * in_factor, dim * out_factor, spatial_stride=2))
                    ops.append(up)
            self.spatial_modulation.append(ops)


    def forward(self, inputs):
        out = []
        for i, feature in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], list):
                out_ = feature
                for _, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](feature))
        return out


class LevelFusion(fluid.dygraph.Layer):

    def __init__(self, in_channels=[1024, 1024], mid_channels=[1024, 1024],
        out_channels=2048, ds_scales=[(1, 1, 1), (1, 1, 1)]):
        super(LevelFusion, self).__init__()

        self.ops = []
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = self.add_sublayer('downsampling_%d' % i, 
                Downsampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1),
                stride=(1, 1, 1), downsample_position='before', downsample_scale=ds_scales[i]))
            self.ops.append(op)
        in_dims = np.sum(mid_channels)
        self.fusion_conv = Conv3DBN(in_dims, out_channels, act='relu')
    

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = fluid.layers.concat(out, 1)
        out = self.fusion_conv(out)
        return out
            

class AuxHead(fluid.dygraph.Layer):

    def __init__(self, inplanes, planes, loss_weight=0.5):
        super(AuxHead, self).__init__()

        self.convs = Conv3DBN(inplanes, inplanes * 2, (1, 3, 3), (1, 2, 2), act='relu')
        self.loss_weight = loss_weight
        self.dropout = fluid.dygraph.Dropout(p=0.5)
        self.fc = Linear(inplanes * 2, planes)
    

    def forward(self, inputs, target=None, for_test=False):
        if target is None:
            return None
        
        loss, accuracy = dict(), dict()
        x = self.convs(inputs) # (n, c, t, h, w)
        x = fluid.layers.adaptive_pool3d(x, 1) # (n, c, 1, 1, 1)
        x = fluid.layers.reshape(x, (x.shape[0], -1)) # (n, c)
        x = self.dropout(x)
        x = self.fc(x) # (n, c)
        loss_aux = self.loss_weight * fluid.layers.softmax_with_cross_entropy(x, target, axis=1)
        loss['loss_aux'] = fluid.layers.reduce_mean(loss_aux)

        if for_test:
            x = fluid.layers.reduce_mean(x, 0, keep_dim=True)
            target = target[0:1,:]

        aux_score = fluid.layers.softmax(x)
        accuracy['aux_acc1'] = fluid.layers.accuracy(input=aux_score, label=target, k=1)
        accuracy['aux_acc5'] = fluid.layers.accuracy(input=aux_score, label=target, k=5)
        return loss, accuracy, aux_score
    