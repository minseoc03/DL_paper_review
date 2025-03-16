import torch
from torch import nn

class WiderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, projection=None, drop_p=0.3):
        super().__init__()

        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias = False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(drop_p),
                                      nn.Conv2d(out_channels, out_channels, 3, padding=1, bias = False))

        self.projection = projection

    def forward(self, x):

        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        out = residual + shortcut
        return out

class WRN(nn.Module):
    def __init__(self, depth, k, num_classes=1000, init_weights=True):
        super().__init__()
        N = int((depth-4)/3/2)

        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias = False)
        self.stage1 = self.make_stage(16*k, N, stride = 1)
        self.stage2 = self.make_stage(32*k, N, stride = 2)
        self.stage3 = self.make_stage(64*k, N, stride = 2)
        self.bn = nn.BatchNorm2d(64*k)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*k, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def make_stage(self, out_channels, num_blocks, stride):

        if stride != 1 or self.in_channels != out_channels:
            projection = nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias = False)
        else:
            projection = None

        layers = []
        layers += [WiderBlock(self.in_channels, out_channels, stride, projection)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers += [WiderBlock(self.in_channels, out_channels)]

        return nn.Sequential(*layers)

if __name__ == "__main__":
  model = WRN(depth=28, k=10, num_classes=10)
  #...
