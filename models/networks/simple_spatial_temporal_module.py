import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from TPN.models.networks.network_libs import AvgPool3D


class SimpleSpatialTemporalModule(fluid.dygraph.Layer):

    def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
        super(SimpleSpatialTemporalModule, self).__init__()
        assert spatial_type == 'avg'

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) \
            else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size, ) + self.spatial_size
        self.pool = AvgPool3D(self.pool_size, 1, 0)
    

    def forward(self, inputs):
        return self.pool(inputs)
