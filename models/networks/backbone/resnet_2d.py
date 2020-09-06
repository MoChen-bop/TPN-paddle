import os 
import sys
sys.path.append('/home/aistudio')
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

from TPN.models.networks.network_libs import ConvBN


class BottleneckBlock(fluid.dygraph.Layer):

    def __init__(self, num_channels, num_filters, stride=1, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBN(num_channels, num_filters, filter_size=1)
        self.conv1 = ConvBN(num_filters, num_filters, filter_size=3)
        self.conv2 = ConvBN(num_filters, num_filters * 4, filter_size=1, act=None)

        if not shortcut:
            self.short = ConvBN(num_channels, num_filters * 4, filter_size=1)
        
        self.shortcut = shortcut
        self.num_channels_out = num_filters * 4
    

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        
        x = fluid.layers.elementwise_add(short, x)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(x)


class ResNet(fluid.dygraph.Layer):

    def __init__(self, depth, in_channels=3, class_dim=1024):
        super(ResNet, self).__init__()

        self.depth = depth
        self.num_channels = [64, 256, 512, 1024]
        self.num_filters = [64, 128, 256, 512]

        self.conv = ConvBN(in_channels, 64, filter_size=7, stride=2)
        self.pool = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(self.depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(num_channels=self.num_channels[block]
                        if i == 0 else self.num_filters[block] * 4,
                        num_filters=self.num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True 
        
        self.pool_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)
        self.feature_dim = self.num_filters[-1] * 4 * 1 * 1

        import math 
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.linear = Linear(self.feature_dim, class_dim, act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))
    

    def forward(self, inputs, require_feature=False):
        x = self.conv(inputs)
        x = self.pool(x)
        for bottleneck_block in self.bottleneck_block_list:
            x = bottleneck_block(x)
        x = self.pool_avg(x)
        features = fluid.layers.reshape(x, shape=[-1, self.feature_dim])

        if require_feature:
            return features
        else:
            logit = self.linear(features)
            return logit


def ResNet50(in_channels=3, class_dim=1024):
    depth = [3, 4, 6, 3]
    return ResNet(depth, in_channels, class_dim)


def ResNet101(in_channels=3, class_dim=1024):
    depth = [3, 4, 23, 3]
    return ResNet(depth, in_channels, class_dim)


def ResNet150(in_channels=3, class_dim=1024):
    depth = [3, 8, 36, 3]
    return ResNet(depth, in_channels, class_dim)


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = ResNet50()
        img = np.zeros([32, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        while True:
            feature = network(img, require_feature=True)
            print(feature.shape) # (1, 2048)
        # print(logit.shape) # (1, 1024)