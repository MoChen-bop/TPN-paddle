import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv3D, BatchNorm, Conv2D


class Conv3DBN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, filter_size=1, stride=1, act=None):
        super(Conv3DBN, self).__init__()
        padding = filter_size // 2 if isinstance(filter_size, int) else (filter_size[0] // 2, filter_size[1] // 2, filter_size[2] // 2)
        self.conv = Conv3D(in_channels, out_channels, filter_size=filter_size,
            stride=stride, padding=padding, bias_attr=False)
        self.bn = BatchNorm(out_channels, act)


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)

        return x


class Conv3x3x3(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, dilation=1):
        super(Conv3x3x3, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=dilation, dilation=dilation, bias_attr=False)
    

    def forward(self, inputs):
        return self.conv(inputs)


class Conv1x1x1(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, dilation=1):
        super(Conv1x1x1, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=1,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=0, dilation=dilation, bias_attr=False)
    

    def forward(self, inputs):
        return self.conv(inputs)


class Conv1x3x3(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, dilation=1):
        super(Conv1x3x3, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=(1, 3, 3),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(0, dilation, dilation), dilation=dilation, bias_attr=False)


    def forward(self, inputs):
        return self.conv(inputs)


class Conv3x1x1(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, dilation=1):
        super(Conv3x1x1, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=(3, 1, 1),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(1, 0, 0), dilation=dilation, bias_attr=False)


    def forward(self, inputs):
        return self.conv(inputs)


class Conv1x1x1BN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, 
        dilation=1, act=None):
        super(Conv1x1x1BN, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=1,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=0, dilation=dilation, bias_attr=False)
        self.bn = BatchNorm(out_channels, act=act)    


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x

        
class Conv3x3x3BN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1,
        dilation=1, act=None):
        super(Conv3x3x3BN, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(1, dilation, dilation), dilation=dilation, bias_attr=False)
        self.bn = BatchNorm(out_channels, act=act)    


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x

        
class Conv1x3x3BN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, 
        dilation=1, act=None):
        super(Conv1x3x3BN, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=(1, 3, 3),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(0, dilation, dilation), dilation=dilation, bias_attr=False)
        self.bn = BatchNorm(out_channels, act=act)    


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x


class Conv3x1x1BN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, spatial_stride=1, temporal_stride=1, 
        dilation=1, act=None):
        super(Conv3x1x1BN, self).__init__()
        self.conv = Conv3D(in_channels, out_channels, filter_size=(3, 1, 1),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(1, 0, 0), dilation=dilation, bias_attr=False)
        self.bn = BatchNorm(out_channels, act=act)    


    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x


class MaxPool3D(fluid.dygraph.Layer):

    def __init__(self, pooling_size, stride, padding=None):
        super(MaxPool3D, self).__init__()

        self.pooling_size = pooling_size
        self.stride = stride
        if padding is None:
            self.padding = pooling_size // 2 if isinstance(pooling_size, int) else \
                (pooling_size[0] // 2, pooling_size[1] // 2, pooling_size[2] // 2)
        else:
            self.padding = padding


    def forward(self, inputs):
        x = fluid.layers.pool3d(inputs, pool_size=self.pooling_size, pool_type='max', 
            pool_stride=self.stride, pool_padding=self.padding)

        return x


class AvgPool3D(fluid.dygraph.Layer):

    def __init__(self, pooling_size, stride, padding=None):
        super(AvgPool3D, self).__init__()

        self.pooling_size = pooling_size
        self.stride = stride
        if padding is None:
            self.padding = pooling_size // 2 if isinstance(pooling_size, int) else \
                (pooling_size[0] // 2, pooling_size[1] // 2, pooling_size[2] // 2)
        else:
            self.padding = padding


    def forward(self, inputs):
        x = fluid.layers.pool3d(inputs, pool_size=self.pooling_size, pool_type='avg', 
            pool_stride=self.stride, pool_padding=self.padding)

        return x


class SeparateConv3DBN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=None):
        super(SeparateConv3DBN, self).__init__()

        padding = kernel_size // 2 if isinstance(kernel_size, int) else \
            (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.depthwise = Conv3D(in_channels, in_channels, filter_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.pointwise = Conv3D(in_channels, out_channels, filter_size=1,
            stride=1, padding=0, groups=1)
        self.bn = BatchNorm(out_channels, act=act)
    

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return x



class ConvBN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, filter_size=3, stride=1,
        groups=1, act='relu'):
        super(ConvBN, self).__init__()

        self.conv = Conv2D(num_channels=in_channels, num_filters=out_channels,
            filter_size=filter_size, stride=stride, padding=(filter_size - 1) // 2,
            groups=groups, act=None, bias_attr=False)
        self.bn = BatchNorm(num_channels=out_channels, act=act)
    

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x


class SeparateConvBN(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, filter_size=1, stride=1, dilation=1, act=None):
        super(SeparateConvBN, self).__init__()

        self.conv = Conv2D(num_channels=in_channels, num_filters=in_channels,
            filter_size=filter_size, stride=stride, padding=(filter_size // 2) * dilation,
            groups=in_channels, dilation=dilation)
        self.pointwise = Conv2D(num_channels=in_channels, num_filters=out_channels,
            filter_size=1, stride=1, padding=0, groups=1, dilation=1)
        self.bn = BatchNorm(out_channels, act=act)
    

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return x





