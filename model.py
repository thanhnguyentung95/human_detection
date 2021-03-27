from torch import nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, 
            groups=1, bias=True, padding_mode='zeros', activation='leaky'):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if activation == 'leakly':
            self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif activation == 'linear':
            pass

    def forward():
        



class Yolov4Tiny(nn.Module):
    def __init__(self):
        super(Yolov4Tiny, self).__init__()

        self.model = nn.Sequential(
            Conv2d(3, 32, 3, stride=2, padding=1)
        )