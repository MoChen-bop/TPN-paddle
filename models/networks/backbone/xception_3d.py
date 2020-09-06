import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from TPN.models.networks.network_libs import Conv3DBN, SeparateConv3DBN
from TPN.models.networks.network_libs import Conv1x3x3BN


def check_data(data, number):
    if len(data) == 1:
        return data * number
    assert len(data) == number
    return data


class XceptionBlock(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reps=3):
        super(XceptionBlock, self).__init__()

        self.skip = None
        if out_channels != in_channels or stride != 1:
            self.skip = Conv3DBN(in_channels, out_channels, kernel_size, stride=stride)
        
        self.stride = stride
        rep = []
        channels = out_channels
        rep.append(('conv_0', SeparateConv3DBN(in_channels, channels, kernel_size, 1, act='relu')))

        for i in range(1, reps + 1):
            rep.append(('conv_%d' % i, SeparateConv3DBN(channels, channels, kernel_size, 1, act='relu')))
        
        self.rep = fluid.dygraph.Sequential(*rep)
        self.pool_stride = stride
        self.pool_size = kernel_size
        self.pool_padding = kernel_size // 2 if isinstance(kernel_size, int) else \
            (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    

    def forward(self, inputs):
        dx = self.rep(inputs)
        dx = self.pool(dx)
        x = self.shortcut(inputs)

        return x + dx
    

    def pool(self, x):
        if self.stride != 1:
            x = fluid.layers.pool3d(x, self.pool_size, pool_stride=self.pool_stride, 
                pool_padding=self.pool_padding, pool_type='max')
        return x
    

    def shortcut(self, x):
        if self.skip is not None:
            x = self.skip(x)
        return x


class Xception(fluid.dygraph.Layer):
    arch_setting = {
        41: {
            'entry_flow': (3, [(1, 3, 3)], [(1, 2, 2)], [128, 256, 728]),
            'middle_flow': (8, [(1, 3, 3)], [1], [728]),
            'exit_flow': (2, [(3, 1, 1), (1, 3, 3)], [(1, 2, 2), 1], [[1024, 1024, 1024], [1536, 1536, 2048]])
        },
        64: {
            'entry_flow': (3, [(1, 3, 3)], [(1, 2, 2)], [128, 256, 728]),
            'middle_flow': (16, [(1, 3, 3)], [1], [728]),
            'exit_flow': (2, [(3, 1, 1), (1, 3, 3)], [(1, 2, 2), 1], [[1024, 1024, 1024], [1536, 1536, 2048]])
        },
        71: {
            'entry_flow': (5, [(1, 3, 3)], [(1, 2, 2), 1, (1, 2, 2), 1, (1, 2, 2)], [128, 256, 256, 728, 728]),
            'middle_flow': (16, [(1, 3, 3)], [1], [728]),
            'exit_flow': (2, [(3, 1, 1), (1, 3, 3)], [(1, 2, 2), 1], [[1024, 1024, 1024], [1536, 1536, 2048]])
        }
    }

    def __init__(self, depth, **kwargs):
        super(Xception, self).__init__()
        setting = self.arch_setting[depth]

        self.convbn1 = Conv1x3x3BN(3, 32, 2, 1, act='relu')
        self.convbn2 = Conv1x3x3BN(32, 64, 1, 1, act='relu')

        self.convbn3 = Conv1x3x3BN(728, 1024, 1, 1, act='relu')

        in_channels = 64
        self.entry_flow, in_channels = self.block_flow(in_channels, *setting['entry_flow'])
        self.middle_flow, in_channels = self.block_flow(in_channels, *setting['middle_flow'])
        in_channels = 1024
        self.exit_flow, in_channels = self.exit_block_flow(in_channels, *setting['exit_flow'])



    def forward(self, inputs):
        x = self.convbn1(inputs) # (n,  32, 112, 112)
        x = self.convbn2(x)      # (n,  64, 112, 112)
        x = self.entry_flow(x)   # (n, 728,  14,  14)
        x = self.middle_flow(x)  # (n,  728, 14, 14)
        
        out_1 = self.convbn3(x)       # (n, 1024, 14, 14)
        out_2 = self.exit_flow(out_1) # (n, 2048,  7,  7)

        return (out_1, out_2)


    def block_flow(self, in_channels, block_num, kernel_sizes, strides, chns):
        kernel_sizes = check_data(kernel_sizes, block_num)
        strides = check_data(strides, block_num)
        chns = check_data(chns, block_num)
        blocks = []
        for i in range(block_num):
            out_channels = chns[i]
            blocks.append(('block_%d' % i,
                XceptionBlock(in_channels, out_channels, kernel_sizes[i], strides[i], 3)))
            in_channels = out_channels
        return fluid.dygraph.Sequential(*blocks), in_channels


    def exit_block_flow(self, in_channels, block_num, kernel_sizes, strides, chns):
        assert block_num == 2

        blocks = []
        blocks.append(('block_0_0', XceptionBlock(in_channels, chns[0][0], kernel_sizes[0], strides[0], 1)))
        blocks.append(('block_0_1', XceptionBlock(chns[0][0], chns[0][1], kernel_sizes[1], strides[1], 1)))
        blocks.append(('block_0_2', XceptionBlock(chns[0][1], chns[0][2], kernel_sizes[1], strides[1], 1)))

        blocks.append(('block_1_0', XceptionBlock(chns[0][2], chns[1][0], kernel_sizes[0], strides[1], 1)))
        blocks.append(('block_1_1', XceptionBlock(chns[1][0], chns[1][1], kernel_sizes[1], strides[1], 1)))
        blocks.append(('block_1_2', XceptionBlock(chns[1][1], chns[1][2], kernel_sizes[1], strides[1], 1)))

        return fluid.dygraph.Sequential(*blocks), chns[1][2]


if __name__ == '__main__':
    import numpy as np
    from paddle.fluid.dygraph import to_variable

    with fluid.dygraph.guard():

        xception = Xception(64)
        frames = to_variable(np.zeros((1, 3, 32, 224, 224)).astype('float32'))
        while True:
            out = xception(frames)
            for o in out:
                print(o.shape)
            # [1, 1024, 8, 14, 14]
            # [1, 2048, 8, 7, 7]


