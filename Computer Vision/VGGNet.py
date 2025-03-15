import torch
from torch import nn

cfgs = { "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
         "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
         "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
         "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"] }

class VGG(nn.Module):
    def __init__(self, cfg, batch_norm, num_classes = 1000, init_weights = True, drop_p = 0.5):
        super().__init__()

        self.features = self.make_layers(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=drop_p),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=drop_p),
                                        nn.Linear(4096, num_classes))

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg, batch_norm = False):
        layers = []
        in_channels = 3
        for v in cfg:
            if type(v) == int:
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, v, 3, padding=1),
                               nn.BatchNorm2d(v),
                               nn.ReLU()]
                else:
                    layers += [nn.Conv2d(in_channels, v, 3, padding=1),
                               nn.ReLU()]
                in_channels = v
            else:
                layers += [nn.MaxPool2d(2)]

        return nn.Sequential(*layers)

if __name__ == "__main__":
  model = VGG(cfgs["E"], batch_norm=True)
  #...
