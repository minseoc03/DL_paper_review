import torch
from torch import nn

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, in_channels, inner_channels, cardinality, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, groups = cardinality, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels * self.expansion))
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        out = self.relu(residual + shortcut)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, num_block_list, cardinality, num_classes = 1000, zero_init_residual = True):
        super().__init__()

        self.in_channels = 64
        self.cardinality = cardinality

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.make_stage(block, 128, num_block_list[0], stride=1)
        self.stage2 = self.make_stage(block, 256, num_block_list[1], stride=2)
        self.stage3 = self.make_stage(block, 512, num_block_list[2], stride=2)
        self.stage4 = self.make_stage(block, 1024, num_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)

    def make_stage(self, block, inner_channels, num_blocks, stride = 1):

        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion))
        else:
            projection = None

        layers = []
        layers += [block(self.in_channels, inner_channels, self.cardinality, stride, projection)]
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks):
            layers += [block(self.in_channels, inner_channels, self.cardinality)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
  def resnext50(**kwargs):
    return ResNeXt(Bottleneck, [3, 4, 6, 3], cardinality=32, **kwargs)

  def resnext101(**kwargs):
    return ResNeXt(Bottleneck, [3, 4, 23, 3], cardinality=32, **kwargs)

  model = resnext50()
  #...
