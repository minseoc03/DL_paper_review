import torch
from torch import nn

class DepSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,in_channels,3, stride = stride, padding = 1, groups = in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True))

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels,out_channels,1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, alpha, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, int(32*alpha), 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*alpha))
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = DepSepConv(int(32*alpha), int(64*alpha), stride=1)
        self.conv3 = nn.Sequential(DepSepConv(int(64*alpha), int(128*alpha), stride=2),
                                   DepSepConv(int(128*alpha), int(128*alpha)))
        self.conv4 = nn.Sequential(DepSepConv(int(128*alpha), int(256*alpha), stride=2),
                                   DepSepConv(int(256*alpha), int(256*alpha)))
        self.conv5 = nn.Sequential(DepSepConv(int(256*alpha), int(512*alpha), stride=2),
                                   *[DepSepConv(int(512*alpha), int(512*alpha)) for i in range(5)])
        self.conv6 = nn.Sequential(DepSepConv(int(512*alpha), int(1024*alpha), stride=2),
                                   DepSepConv(int(1024*alpha), int(1024*alpha)))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(1024*alpha), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
  model = MobileNet(alpha=1)
  #...
