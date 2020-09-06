import sys
sys.path.append('/home/aistudio')
import paddle
import paddle.fluid as fluid

from TPN.models.networks.network_libs import Conv1x1x1BN, Conv3x3x3BN
from TPN.models.networks.network_libs import Conv1x3x3BN, Conv3x1x1BN
from TPN.models.networks.network_libs import Conv3DBN, MaxPool3D, SeparateConv3DBN


class XBottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride=1,
        dilation=1, if_inflate=True, inflate_style='3x1x1'):
        super(XBottleneck, self).__init__()
        assert inflate_style in ['3x1x1', '3x3x3']

        if if_inflate:
            if inflate_style == '3x1x1':
                self.conv1 = SeparateConv3DBN(inplanes, planes, (3, 1, 1), act='relu')
                self.conv2 = SeparateConv3DBN(planes, planes, (1, 3, 3), (temporal_stride, spatial_stride, spatial_stride), act='relu')
            else:
                self.conv1 = Conv1x1x1BN(inplanes, planes, act='relu')
                self.conv2 = Conv3x3x3BN(planes, planes, spatial_stride, temporal_stride, act='relu')
        else:
            self.conv1 = SeparateConv3DBN(inplanes, planes, (1, 1, 1), act='relu')
            self.conv2 = SeparateConv3DBN(planes, planes, (1, 3, 3), (temporal_stride, spatial_stride, spatial_stride), act='relu')

        self.conv3 = SeparateConv3DBN(planes, planes * self.expansion, 1)

        self.downsample = None
        if inplanes != planes * self.expansion:
            self.downsample = Conv1x1x1BN(inplanes, planes * self.expansion, spatial_stride, temporal_stride)


    def forward(self, inputs):
        identity = inputs

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        out = fluid.layers.relu(x)

        return out
    

class XResNet3D(fluid.dygraph.Layer):
    arch_settings = {
        50: (XBottleneck, (3, 4, 6, 3)),
        101: (XBottleneck, (3, 4, 23, 3)),
        152: (XBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, num_stages=4, spatial_strides=(1, 2, 2, 2), temporal_strides=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3),
        conv1_kernel_t=1, conv1_stride_t=1, pool1_kernel_t=1, pool1_stride_t=1,
        inflate_freqs=(1, 1, 1, 1), inflate_style='3x1x1'):
        super(XResNet3D, self).__init__()

        self.depth = depth
        self.num_stages = num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.inflate_freqs = inflate_freqs if not isinstance(inflate_freqs, int) else (inflate_freqs) * num_stages
        self.inflate_style = inflate_style

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.conv1 = Conv3DBN(3, 64, filter_size=(conv1_kernel_t, 7, 7), stride=(conv1_stride_t, 2, 2), act='relu')
        self.pool1 = MaxPool3D((pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2))
        
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i

            res_layer = self.make_res_layer(i, self.block, self.inplanes, planes,
                num_blocks, spatial_stride, temporal_stride, dilation, inflate_freqs[i],
                self.inflate_style)
            
            self.inplanes = planes * self.block.expansion   
            self.res_layers.append(res_layer)
        
        self.feature_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)
    

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        outs = []
        for i, layer in enumerate(self.res_layers):
            x = layer(x)

            if i in self.out_indices:
                outs.append(x)
        
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


    def make_res_layer(self, block_id, block, inplanes, planes, blocks, spatial_stride=1,
        temporal_stride=1, dilation=1, inflate_freq=1, inflate_style='3x1x1'):
        inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else [inflate_freq] * blocks
        
        layers = []
        layers.append(('resblock_%d_0' % block_id, block(inplanes, planes, spatial_stride, temporal_stride,
            dilation, if_inflate=(inflate_freq[0] == 1), inflate_style=inflate_style, )))
       
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(('resblock_%d_%d' % (block_id, i), 
                block(inplanes, planes, 1, 1, dilation, 
                if_inflate=(inflate_freq[i] == 1), inflate_style=inflate_style)))
        
        layers = fluid.dygraph.Sequential(*layers)
        return layers


if __name__ == '__main__':
    import numpy as np
    from paddle.fluid.dygraph import to_variable

    with fluid.dygraph.guard():

        resnet3d = XResNet3D(50, inflate_freqs=(0, 0, 1, 1))
        frames = to_variable(np.zeros((1, 3, 32, 224, 224)).astype('float32'))
        while True:
            out = resnet3d(frames)
            for o in out:
                print(o.shape)
        # [4, 256, 8, 56, 56]
        # [4, 512, 8, 28, 28]
        # [4, 1024, 8, 14, 14]
        # [4, 2048, 8, 7, 7]

