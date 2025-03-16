import torch
from torch import nn

class DepSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,in_channels,3, stride = stride, padding = 1, groups = in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU6(inplace=True))

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels,out_channels,1, bias=False),
                                       nn.BatchNorm2d(out_channels))
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class InvertedBlock(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, stride):
        super().__init__()

        self.use_skip_connect = (stride==1 and in_channels==out_channels)

        layers = []
        if in_channels != exp_channels:
            layers += [nn.Sequential(nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                                     nn.BatchNorm2d(exp_channels),
                                     nn.ReLU6(inplace=True))]
        layers += [DepSepConv(exp_channels, out_channels, stride=stride)]

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connect:
            return x + self.residual(x)
        else:
            return self.residual(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.configs=[# t, c, n, s
                      [1, 16, 1, 1],
                      [6, 24, 2, 2],
                      [6, 32, 3, 2],
                      [6, 64, 4, 2],
                      [6, 96, 3, 1],
                      [6, 160, 3, 2],
                      [6, 320, 1, 1]]

        self.stem_conv = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU6(inplace=True))

        in_channels = 32
        layers = []
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                exp_channels = in_channels * t
                layers += [InvertedBlock(in_channels=in_channels, exp_channels=exp_channels, out_channels=c, stride=stride)]
                in_channels = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, 1280, 1, bias=False),
                                       nn.BatchNorm2d(1280),
                                       nn.ReLU6(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(1280, num_classes))

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
  model = MobileNetV2()
  #...
