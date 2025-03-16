import torch
from torch import nn
from torchvision import transforms, datasets
from torchvision.ops import StochasticDepth
import math

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
    def __init__(self, in_channels, squeeze_channels): 
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(nn.Linear(in_channels, squeeze_channels),
                                        nn.SiLU(inplace=True),
                                        nn.Linear(squeeze_channels, in_channels),
                                        nn.Sigmoid())

    def forward(self, x):
        SE = self.squeeze(x)
        SE = SE.reshape(x.shape[0],x.shape[1])
        SE = self.excitation(SE)
        SE = SE.unsqueeze(dim=2).unsqueeze(dim=3)
        x = x * SE
        return x

class DepSESep(nn.Module):
    def __init__(self, in_channels, squeeze_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride = stride, padding = (kernel_size - 1) // 2, groups = in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels, momentum=0.99, eps=1e-3),
                                       nn.SiLU(inplace=True))

        self.seblock = SEBlock(in_channels, squeeze_channels)

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                       nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3))
                                      
    def forward(self, x):
        x = self.depthwise(x)
        if self.seblock is not None:
            x = self.seblock(x)
        x = self.pointwise(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, sd_prob):
        super().__init__()

        self.use_skip_connect = (stride==1 and in_channels==out_channels)
        self.stochastic_depth = StochasticDepth(sd_prob, "row")

        layers = []
        if in_channels != exp_channels:
            layers += [nn.Sequential(nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                                     nn.BatchNorm2d(exp_channels, momentum=0.99, eps=1e-3),
                                     nn.SiLU(inplace=True))]
        squeeze_channels = in_channels // 4
        layers += [DepSESep(exp_channels, squeeze_channels, out_channels, kernel_size, stride=stride)]

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connect:
            residual = self.residual(x)
            residual = self.stochastic_depth(residual)
            return x + residual
        else:
            return self.residual(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes, depth_mult, width_mult, resize_size, crop_size, drop_p, stochastic_depth_p = 0.2):
        super().__init__()

        cfgs = [#k,  t,   c,  n,  s
                [3,  1,  16,  1,  1],
                [3,  6,  24,  2,  2],
                [5,  6,  40,  2,  2],
                [3,  6,  80,  3,  2],
                [5,  6,  112, 3,  1],
                [5,  6,  192, 4,  2],
                [3,  6,  320, 1,  1]]

        in_channels = _make_divisible(32 * width_mult, 8)

        self.transforms = transforms.Compose([transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                                              transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.stem_conv = nn.Sequential(nn.Conv2d(3, in_channels, 3, padding=1, stride=2, bias=False),
                                       nn.BatchNorm2d(in_channels, momentum=0.99, eps=1e-3),
                                       nn.SiLU(inplace=True))

        layers = []
        num_block = 0
        N = sum([math.ceil(cfg[-2] * depth_mult) for cfg in cfgs])
        for k, t, c, n, s in cfgs:
            n = math.ceil(n * depth_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                exp_channels = _make_divisible(in_channels * t, 8)
                out_channels = _make_divisible(c * width_mult, 8)
                sd_prob = stochastic_depth_p * num_block / (N-1)
                layers += [MBConv(in_channels, exp_channels, out_channels, k, stride, sd_prob)]
                in_channels = out_channels
                num_block += 1

        self.layers = nn.Sequential(*layers)

        last_channels = _make_divisible(1280 * width_mult, 8)
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, last_channels, 1, bias=False),
                                       nn.BatchNorm2d(last_channels, momentum=0.99, eps=1e-3),
                                       nn.SiLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(nn.Dropout(drop_p),
                                        nn.Linear(last_channels, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / torch.sqrt(torch.tensor(m.out_features))
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
  def efficientnet_b0(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=1.0, width_mult=1.0, resize_size=256, crop_size=224, drop_p=0.2, **kwargs)
  
  def efficientnet_b1(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=1.1, width_mult=1.0, resize_size=256, crop_size=240, drop_p=0.2, **kwargs)
  
  def efficientnet_b2(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=1.2, width_mult=1.1, resize_size=288, crop_size=288, drop_p=0.3, **kwargs)
  
  def efficientnet_b3(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=1.4, width_mult=1.2, resize_size=320, crop_size=300, drop_p=0.3, **kwargs)
  
  def efficientnet_b4(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=1.8, width_mult=1.4, resize_size=384, crop_size=380, drop_p=0.4, **kwargs)
  
  def efficientnet_b5(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=2.2, width_mult=1.6, resize_size=456, crop_size=456, drop_p=0.4, **kwargs)
  
  def efficientnet_b6(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=2.6, width_mult=1.8, resize_size=528, crop_size=528, drop_p=0.5, **kwargs)
  
  def efficientnet_b7(num_classes=1000, **kwargs):
      return EfficientNet(num_classes=num_classes, depth_mult=3.1, width_mult=2.0, resize_size=600, crop_size=600, drop_p=0.5, **kwargs)

  model = efficientnet_b7()
  #...
