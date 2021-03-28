from torch import nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, 
            groups=1, padding_mode='zeros', activation=None, batchnorm=False):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride=stride, 
                                padding=padding,
                                bias=not batchnorm)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)  # same hyperparameters with Scaled Yolo v4
        elif activation == None:  # no activation 
            pass

        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)  # same hyperparameters with Scaled Yolo v4

    def forward():
        pass


class Yolov4Tiny(nn.Module):
    def __init__(self):
        ''' Refer to ./cfg/yolov4-tiny.cfg '''
        super(Yolov4Tiny, self).__init__()

        self.model = nn.Sequential(
            Conv2d(3, 32, 3, stride=2, padding=1, activation='leaky', batchnorm=True), # 25
            Conv2d(32, 64, 3, stride=2, padding=1, activation='leaky', batchnorm=True),  # 33
            # 
        )