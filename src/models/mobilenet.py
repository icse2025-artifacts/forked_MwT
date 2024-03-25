import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
DEFAULT_MOBILENET_CONV_CFG = [
    [3, 32, -1, None],
    # Format: (in_channels, out_channels, stride, depthwise/pointwise?)
    [32, 32, 1, True], [32, 64, 1, False],
    [64, 64, 2, True], [64, 128, 1, False],
    [128, 128, 1, True], [128, 128, 1, False],
    [128, 128, 2, True], [128, 256, 1, False],
    [256, 256, 1, True], [256, 256, 1, False],
    [256, 256, 2, True], [256, 512, 1, False],
    [512, 512, 1, True], [512, 512, 1, False],
    [512, 512, 1, True], [512, 512, 1, False],
    [512, 512, 1, True], [512, 512, 1, False],
    [512, 512, 1, True], [512, 512, 1, False],
    [512, 512, 1, True], [512, 512, 1, False],
    [512, 512, 2, True], [512, 1024, 1, False],
    [1024, 1024, 1, True], [1024, 1024, 1, False],
]


class MobileNet(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes

        if not conv_configs:
            conv_configs = DEFAULT_MOBILENET_CONV_CFG

        self.conv_configs = conv_configs

        self.block_count = None
        self._make_blocks()

        self.fc = nn.Linear(conv_configs[-1][1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_blocks(self):
        self.block_count = 0

        in_channel, out_channel, _, _ = self.conv_configs[0]
        setattr(self, f'conv_block_0', nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        ))
        self.block_count += 1

        # make blocks
        for i in range(1, len(self.conv_configs) - 1, 2):
            layers = []
            config_dw, config_pw = self.conv_configs[i], self.conv_configs[i + 1]
            for config in [config_dw, config_pw]:
                in_channels, out_channels, stride, is_depthwise = config
                if is_depthwise:
                    layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                         padding=1, groups=in_channels, bias=False),
                               nn.BatchNorm2d(in_channels),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                         padding=0, bias=False),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(inplace=True)]
            setattr(self, f'conv_block_{self.block_count}', nn.Sequential(*layers))
            self.block_count += 1

    def forward(self, x):
        out = self.conv_block_0(x)
        # Iterate over the remaining blocks
        for i in range(1, self.block_count):
            block_i = getattr(self, f'conv_block_{i}')
            out = block_i(out)

        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    from src.model_utils import print_model_summary

    model = MobileNet(num_classes=10)
    print_model_summary(model)
    images = torch.rand(1, 3, 32, 32)
    output = model(images)
    print(output)
