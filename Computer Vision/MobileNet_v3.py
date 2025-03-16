import torch
from torch import nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

class SEBlock(nn.Module):
    def __init__(self, in_channels, r = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(nn.Linear(in_channels, _make_divisible(in_channels // r, 8)),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(_make_divisible(in_channels // r, 8), in_channels),
                                        nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        SE = self.squeeze(x)
        SE = SE.reshape(x.shape[0],x.shape[1])
        SE = self.excitation(SE)
        SE = SE.unsqueeze(dim=2).unsqueeze(dim=3)
        x = x * SE
        return x

class DepSESep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_se, use_hs, stride):
        super().__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride = stride, padding = (kernel_size - 1) // 2, groups = in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels, momentum=0.99), 
                                       nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True))

        self.seblock = SEBlock(in_channels) if use_se else None

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels,1, bias=False),
                                       nn.BatchNorm2d(out_channels, momentum=0.99))

    def forward(self, x):
        x = self.depthwise(x)
        if self.seblock is not None:
            x = self.seblock(x)
        x = self.pointwise(x)
        return x

class InvertedBlock(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, use_hs):
        super().__init__()

        self.use_skip_connect = (stride==1 and in_channels==out_channels)

        layers = []
        if in_channels != exp_channels: 
            layers += [nn.Sequential(nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                                     nn.BatchNorm2d(exp_channels, momentum=0.99),
                                     nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True))]
        layers += [DepSESep(exp_channels, out_channels, kernel_size, use_se, use_hs, stride=stride)]

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connect:
            return x + self.residual(x) 
        else:
            return self.residual(x)

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, last_channels, num_classes=1000, width_mult=1.):
        super().__init__()

        in_channels = _make_divisible(16 * width_mult, 8)

        self.stem_conv = nn.Sequential(nn.Conv2d(3, in_channels, 3, padding=1, stride=2, bias=False),
                                       nn.BatchNorm2d(in_channels, momentum=0.99),
                                       nn.Hardswish(inplace=True))

        layers=[]
        for k, t, c, use_se, use_hs, s in cfgs:
            exp_channels = _make_divisible(in_channels * t, 8)
            out_channels = _make_divisible(c * width_mult, 8)
            layers += [InvertedBlock(in_channels, exp_channels, out_channels, k, s, use_se, use_hs)]
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                                       nn.BatchNorm2d(exp_channels, momentum=0.99),
                                       nn.Hardswish(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        last_channels = _make_divisible(last_channels * width_mult, 8)
        self.classifier = nn.Sequential(nn.Linear(exp_channels, last_channels),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channels, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenetv3_large(**kwargs):
    cfgs = [#k,   t,   c,   SE,   HS,   s
            [3,   1,  16, False, False, 1],
            [3,   4,  24, False, False, 2],
            [3,   3,  24, False, False, 1],
            [5,   3,  40, True,  False, 2],
            [5,   3,  40, True,  False, 1],
            [5,   3,  40, True,  False, 1],
            [3,   6,  80, False, True,  2],
            [3, 2.5,  80, False, True,  1],
            [3, 2.3,  80, False, True,  1],
            [3, 2.3,  80, False, True,  1],
            [3,   6, 112, True,  True,  1],
            [3,   6, 112, True,  True,  1],
            [5,   6, 160, True,  True,  2],
            [5,   6, 160, True,  True,  1],
            [5,   6, 160, True,  True,  1]]

    return MobileNetV3(cfgs, last_channels=1280, **kwargs)

def mobilenetv3_small(**kwargs):
    cfgs = [#k,    t,   c,  SE,    HS,   s
            [3,    1,  16, True,  False, 2],
            [3,  4.5,  24, False, False, 2],
            [3, 3.67,  24, False, False, 1],
            [5,    4,  40, True,  True,  2],
            [5,    6,  40, True,  True,  1],
            [5,    6,  40, True,  True,  1],
            [5,    3,  48, True,  True,  1],
            [5,    3,  48, True,  True,  1],
            [5,    6,  96, True,  True,  2],
            [5,    6,  96, True,  True,  1],
            [5,    6,  96, True,  True,  1]]

    return MobileNetV3(cfgs, last_channels=1024, **kwargs)

if __name__ == "__main__":
  model = mobilenetv3_large()
  #...
