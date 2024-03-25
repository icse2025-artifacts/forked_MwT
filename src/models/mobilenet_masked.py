import torch
import torch.nn as nn

from models.mobilenet import MobileNet as Std_MobileNet
from models.utils_v2 import MaskConvBN


class MobileNet(Std_MobileNet):
    def __init__(self, num_classes=10, conv_configs=None, keep_generator=True):
        self.keep_generator = keep_generator
        super().__init__(num_classes, conv_configs)

    def _make_blocks(self):
        self.block_count = 0

        # The first conv layer will not be masked.
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
                    layers += [
                        MaskConvBN(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False,
                                   keep_generator=self.keep_generator),
                        nn.ReLU(inplace=True)]
                else:
                    layers += [
                        MaskConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False,
                                   keep_generator=self.keep_generator),
                        nn.ReLU(inplace=True)]
            setattr(self, f'conv_block_{self.block_count}', nn.Sequential(*layers))
            self.block_count += 1

    def get_masks(self):
        masks = []
        for each_module in self.modules():
            if isinstance(each_module, MaskConvBN):
                masks.append(each_module.masks)
        return masks


if __name__ == '__main__':
    from src.model_utils import print_model_summary

    model = MobileNet(num_classes=10)
    print_model_summary(model)
    images = torch.rand(1, 3, 32, 32)
    output = model(images)
    print(output)
