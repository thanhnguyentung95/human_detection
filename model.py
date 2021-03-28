import torch
from torch import nn


class Route(nn.Module):
    def __init__(self, groups=1, group_id=1):
        super(Route, self).__init__()
        self.groups = groups
        self.group_id = group_id

    def forward(self, *input):
        x = torch.cat(input, dim=1)   
        channels = x.shape[1] // self.groups  # number of channels in output
        begin = channels*self.group_id
        out = x[:, begin:begin+channels]
        return 


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=2, padding=1, dilation=1, 
            groups=1, padding_mode='zeros', activation=None, batchnorm=True):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride=stride, 
                                padding=padding,
                                bias=not batchnorm)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)  # same hyperparameters with Scaled Yolo v4
        elif activation == None:  # no activation 
            self.activation = None
        
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)  # same hyperparameters with Scaled Yolo v4
        else:
            self.batchnorm = None

    def forward(self, x):
        x = self.conv2d(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        if self.activation:
            x = self.activation(x)        
        return x


class Yolov4Tiny(nn.Module):
    def __init__(self):
        ''' References ./cfg/yolov4-tiny.cfg '''
        super(Yolov4Tiny, self).__init__()

        # Backbone
        # Conv2d(..., stride=2, padding=1, dilation=1, ..., batchnorm=True)
        self.conv1 = Conv2d(3, 32, 3, stride=2, activation='leaky'),        # 25
        self.conv2 = Conv2d(32, 64, 3, stride=2, activation='leaky'),       # 33
        self.conv3 = Conv2d(64, 64, 3, stride=1, activation='leaky'),       # 41
        self.route1 = Route(groups=2, group_id=1)                           # 49
        self.conv4 = Conv2d(32, 32, 3, stride=1, activation='leaky')        # 54
        self.conv5 = Conv2d(32, 32, 3, stride=1)                            # 62
        self.route2 = Route()                                               # 70
        self.conv6 = Conv2d(64, 64, 1, stride=1, activation='leaky')        # 73
        self.route3 = Route()                                               # 81
        self.maxpool1 = nn.MaxPool2d(2, stride=2)                           # 84
        self.conv7 = Conv2d(128, 128, 3, stride=1, activation='leaky')      # 88
        self.route4 = Route(groups=2, group_id=1)                           # 96
        self.conv8 = Conv2d(64, 64, 3, stride=1, activation='leaky')        # 101
        self.conv9 = Conv2d(64, 64, 3, stride=1, activation='leaky')        # 109
        self.route5 = Route()                                               # 117
        self.conv10 = Conv2d(128, 128, 1, stride=1, activation='leaky')     # 120
        self.route6 = Route()                                               # 128
        self.maxpool2 = nn.MaxPool2d(2, stride=2)                           # 131
        self.conv11 = Conv2d(256, 256, 3, stride=1, activation='leaky')     # 135
        self.route7 = Route(groups=2, group_id=1)                           # 143
        self.conv12 = Conv2d(128, 128, 3, stride=1, activation='leaky')     # 148
        self.conv13 = Conv2d(128, 128, 3, stride=1, activation='leaky')     # 156
        self.route8 = Route()                                               # 164
        self.conv14 = Conv2d(256, 256, 1, stride=1, activation='leaky')     # 167 
        self.route9 = Route()                                               # 175
        self.maxpool3 = nn.MaxPool2d(2, stride=2)                           # 178
        self.conv15 = Conv2d(512, 512, 3, stride=1, activation='leaky')     # 182

        # Neck
        self.conv16 = Conv2d(512, 256, 1, stride=1, activation='leaky')     # 192
        self.conv17 = Conv2d(256, 512, 3, stride=1, activation='leaky')     # 200
        self.conv18 = Conv2d(512, 255, stride=1, activation=None)           # 208