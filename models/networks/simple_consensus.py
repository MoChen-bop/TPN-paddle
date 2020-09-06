import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid


class SimpleConsenus(fluid.dygraph.Layer):

    def __init__(self, consensus_type='avg', dim=1):
        super(SimpleConsenus, self).__init__()
        assert consensus_type == 'avg'
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None
    

    def forward(self, x):
        x = fluid.layers.reduce_mean(x, self.dim, keep_dim=True)
        return x