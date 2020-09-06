import os 
import sys
sys.path.append('/home/aistudio')
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

from TPN.models.networks.network_libs import ConvBN, SeparateConvBN


def check_data(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number  
    return data


class XceptionBlock(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, reps, strides=1):
        super(XceptionBlock, self).__init__()

        self.skip = None
        if out_channels != in_channels or strides != 1:
            self.skip = ConvBN(in_channels, out_channels, filter_size=1, stride=strides, )

        rep = []
        channels = in_channels
        
        rep.append(('conv_1', SeparateConvBN(in_channels, out_channels, 3, 1, act='relu')))
        channels = out_channels
    
        for i in range(reps - 1):
            rep.append(('conv_2_%d' % i, SeparateConvBN(channels, channels, 3, 1, act='relu')))

        if strides != 1:
            rep.append(('max_pool', Pool2D(pool_size=3, pool_type='max', pool_stride=strides, pool_padding=1)))

        self.rep = fluid.dygraph.Sequential(*rep)


    def forward(self, inputs):
        
        x = self.rep(inputs)
        if self.skip is not None:
            skip = self.skip(inputs)
        else:
            skip = inputs
        
        x += skip
        return x


class Xception(fluid.dygraph.Layer):

    def __init__(self, bottleneck_params, in_channels=3, class_dim=1024):
        super(Xception, self).__init__()

        self.convbn1 = ConvBN(in_channels, 32, 3, 2, act='relu')
        self.convbn2 = ConvBN(32, 64, 3, 1, act='relu')

        in_channel = 64
        self.entry_flow, in_channel = self.block_flow(
            block_num=bottleneck_params['entry_flow'][0],
            strides=bottleneck_params['entry_flow'][1],
            chns=bottleneck_params['entry_flow'][2],
            in_channel=in_channel)
            
        self.middle_flow, in_channel = self.block_flow(
            block_num=bottleneck_params['middle_flow'][0],
            strides=bottleneck_params['middle_flow'][1],
            chns=bottleneck_params['middle_flow'][2],
            in_channel=in_channel)
            
        self.exit_flow, in_channel = self.exit_block_flow(
            block_num=bottleneck_params['exit_flow'][0],
            strides=bottleneck_params['exit_flow'][1],
            chns=bottleneck_params['exit_flow'][2],
            in_channel=in_channel)

        self.pool = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        self.feature_dim = 2048

        import math 
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.linear = Linear(self.feature_dim, class_dim, act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def block_flow(self, block_num, strides, chns, in_channel):
        block = []
        strides = check_data(strides, block_num)
        chns = check_data(chns, block_num)
        for i in range(block_num):
            out_channels = chns[i]
            block.append(('block_%d' % i,
                XceptionBlock(in_channel, out_channels, 3, strides[i])))
            in_channel = out_channels

        return fluid.dygraph.Sequential(*block), in_channel
    

    def exit_block_flow(self, block_num, strides, chns, in_channel):
        assert block_num == 2

        block = []
        block.append(('block_0_0',
            XceptionBlock(in_channel, chns[0][0], 1, strides[0][0])))
        block.append(('block_0_1',
            XceptionBlock(chns[0][0], chns[0][1], 1, strides[0][1])))
        block.append(('block_0_2',
            XceptionBlock(chns[0][1], chns[0][2], 1, strides[0][1])))
        
        block.append(('block_1_0',
            XceptionBlock(chns[0][2], chns[1][0], 1, strides[1][0])))
        block.append(('block_1_1',
            XceptionBlock(chns[1][0], chns[1][1], 1, strides[1][1])))
        block.append(('block_1_2',
            XceptionBlock(chns[1][1], chns[1][2], 1, strides[1][1])))
        
        return fluid.dygraph.Sequential(*block), chns[1][2]


    def forward(self, inputs, require_feature=False):
        
        x = self.convbn1(inputs) # (n, 32, 112, 112)
        x = self.convbn2(x) # (n, 64, 112, 112)

        x = self.entry_flow(x) # (n, 728, 14, 14)
        x = self.middle_flow(x) # (n, 728, 14, 14)
        x = self.exit_flow(x) # (n, 2048, 7, 7)

        x = self.pool(x) # (n, 2048, 1, 1)
        features = fluid.layers.reshape(x, shape=[x.shape[0], -1])
        
        if require_feature:
            return features
        else:
            logit = self.linear(features)
            return logit


def Xception41(in_channels=3, class_dim=1024):
    bottleneck_params = {
        'entry_flow':  (3, [2, 2, 2], [128, 256, 728]),
        'middle_flow': (8, 1, 728),
        'exit_flow': (2, [[2, 1, 1], [1, 1, 1]], [[728, 1024, 1024], [1536, 1536, 2048]])
    }
    return Xception(bottleneck_params, in_channels, class_dim)


def Xception64(in_channels=3, class_dim=1024):
    bottleneck_params = {
        'entry_flow':  (3, [2, 2, 2], [128, 256, 728]),
        'middle_flow': (16, 1, 728),
        'exit_flow': (2, [[2, 1, 1], [1, 1, 1]], [[728, 1024, 1024], [1536, 1536, 2048]])
    }
    return Xception(bottleneck_params, in_channels, class_dim)


def Xception71(in_channels=3, class_dim=1024):
    bottleneck_params = {
        'entry_flow':  (5, [2, 1, 2, 1, 2], [128, 256, 256, 728, 728]),
        'middle_flow': (16, 1, 728),
        'exit_flow': (2, [[2, 1, 1], [1, 1, 1]], [[728, 1024, 1024], [1536, 1536, 2048]])
    }
    return Xception(bottleneck_params, in_channels, class_dim)


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = Xception64()
        img = np.zeros([32, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        # feature = network(img, require_feature=True)
        # print(feature.shape) # (1, 2048)
        while True:
            logit = network(img, require_feature=False)
            print(logit.shape) # (1, 1024)
