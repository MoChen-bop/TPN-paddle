import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from TPN.models.networks.network_modules import TemporalModulation
from TPN.models.networks.network_modules import SpatialModulation
from TPN.models.networks.network_modules import Upsampling, Downsampling
from TPN.models.networks.network_modules import LevelFusion, AuxHead
from TPN.models.networks.network_libs import Conv3DBN


class TPN(fluid.dygraph.Layer):

    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256,
        spatial_modulation_config=None, temporal_modulation_config=None, upsampling_config=None,
        downsampling_config=None, level_fusion_config=None, aux_head_config=None):
        super(TPN, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.temporal_modulation_ops = []
        self.upsampling_ops = []
        self.downsampling_ops = []
        self.level_fusion_op = []
        self.spatial_modulation = SpatialModulation(**spatial_modulation_config)

        for i in range(0, self.num_ins, 1):
            inplanes = in_channels[-1]
            planes = out_channels

            temporal_modulation_config.param.inplanes = inplanes
            temporal_modulation_config.param.planes = planes
            temporal_modulation_config.param.downsample_scale = temporal_modulation_config.scales[i]
            temporal_modulation = self.add_sublayer('temporal_down_%d' % i, 
                TemporalModulation(**temporal_modulation_config.param))
            self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_ins - 1:
                upsampling = self.add_sublayer('upsampling_%d' % i, Upsampling(**upsampling_config))
                self.upsampling_ops.append(upsampling)

                downsampling_config.param.inplanes = planes
                downsampling_config.param.planes = planes
                downsampling_config.param.downsample_scale = downsampling_config.scales
                downsampling = self.add_sublayer('downsampling_%d' % i, 
                    Downsampling(**downsampling_config.param))
                self.downsampling_ops.append(downsampling)
        
        self.level_fusion_op = LevelFusion(**level_fusion_config)
        self.level_fusion_ops = LevelFusion(**level_fusion_config)

        out_dims = level_fusion_config.out_channels
        self.pyramid_fusion_op = Conv3DBN(out_dims * 2, 2048, act='relu')
        
        aux_head_config.inplanes = in_channels[-2]
        self.aux_head = AuxHead(**aux_head_config)


    def forward(self, inputs, target=None, for_test=False):
        loss = None

        if target is not None:
            loss, accuracy, aux_score = self.aux_head(inputs[-2], target, for_test)

        outs = self.spatial_modulation(inputs)

        outs = [temporal_modulation(outs[i]) for i, temporal_modulation in enumerate(self.temporal_modulation_ops)]

        temporal_modulation_outs = outs
        
        # top-down
        for i in range(self.num_ins - 1, 0, -1):
            outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])
        topdownouts = self.level_fusion_ops(outs)

        outs = temporal_modulation_outs
        for i in range(0, self.num_ins - 1, 1):
            outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])
        outs = self.level_fusion_op(outs)

        outs = self.pyramid_fusion_op(fluid.layers.concat([topdownouts, outs], 1))

        return outs, loss, accuracy, aux_score


if __name__ == '__main__':
    import numpy as np
    from paddle.fluid.dygraph import to_variable

    from TPN.utils.config import cfg

    with fluid.dygraph.guard():
        tpn = TPN(**cfg.models.tpn)

        features = [to_variable(np.zeros((1, 256, 64, 56, 56)).astype('float32')),
            to_variable(np.zeros((1, 512, 64, 28, 28)).astype('float32')),
            to_variable(np.zeros((1, 1024, 64, 14, 14)).astype('float32')),
            to_variable(np.zeros((1, 2048, 64, 7, 7)).astype('float32')),]
        target = to_variable(np.zeros((1, 1)).astype('int64'))

        outs, loss = tpn(features[-2:], target)
        print(outs.shape)
        print(loss['loss_aux'].shape)
        # [1, 2048, 2, 7, 7]
        # [1, 1]
