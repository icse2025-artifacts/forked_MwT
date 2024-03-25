import copy
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F

from models.mobilenet import DEFAULT_MOBILENET_CONV_CFG
from models.mobilenet_masked import MobileNet as mt_MobileNet_model

from model_utils import print_model_summary

from models.utils_v2 import MaskConvBN
from torch import nn


class ChannelPaddingLayer(nn.Module):
    def __init__(self, in_channels_mask, out_channels_mask):
        super().__init__()
        self.conv_out_channels, self.conv_forward_indices = self.calculate_forward_mask(in_channels_mask,
                                                                                        out_channels_mask)

    @staticmethod
    def calculate_forward_mask(in_channels_mask, out_channels_mask):
        # we can do this way because out_channels_mask contains in_channels_mask
        # check function: get_module_cfg() - LINE: module_layer_masks[i - 1] = each_layer_mask & module_layer_masks[i - 1]
        conv_forward_mask = torch.zeros(out_channels_mask.shape, dtype=torch.bool).cuda()
        conv_forward_mask[in_channels_mask] = True
        actual_conv_forward_mask = conv_forward_mask[out_channels_mask]
        actual_conv_forward_indices = actual_conv_forward_mask.nonzero().squeeze(1)
        actual_conv_out_channels = actual_conv_forward_mask.shape[0]
        return actual_conv_out_channels, actual_conv_forward_indices

    def forward(self, x):
        out = torch.zeros(
            (x.shape[0], self.conv_out_channels, x.shape[2], x.shape[3]),
            dtype=x.dtype, device=x.device
        )
        out[:, self.conv_forward_indices] = x
        return out


class MobileNet_Module(mt_MobileNet_model):
    def __init__(self, num_classes=10, conv_configs=None, keep_generator=True):
        super().__init__(num_classes, conv_configs, keep_generator)
        self.module_head = None

    def build_forward_padding_layer(self, module_layer_masks):
        # calculate mask to handle skip connection during forward pass
        module_layer_masks = module_layer_masks[1:]  # remove the pseudo mask of the first conv layer
        mask_conv_layers = [(layer_name, layer) for layer_name, layer in self.named_modules() if
                            isinstance(layer, MaskConvBN)]
        for layer_index, (layer_name, layer) in enumerate(mask_conv_layers):
            if not re.match(r"conv_block_\d+.0", layer_name):
                continue
            out_channels_mask = module_layer_masks[layer_index].bool()
            if layer_index == 0:
                padding_layer = ChannelPaddingLayer(out_channels_mask, out_channels_mask)
            else:
                in_channels_mask = module_layer_masks[layer_index - 1].bool()
                padding_layer = ChannelPaddingLayer(in_channels_mask, out_channels_mask)

            curr_block_name = layer_name.rsplit(".", 1)[0]

            curr_sequential_module = getattr(self, curr_block_name)
            curr_sequential_module._modules = OrderedDict(list([('-1', padding_layer)]) +
                                                          list(curr_sequential_module._modules.items()))

    def forward(self, x):
        out = super().forward(x)
        out = self.module_head(out)
        return out


def get_module_cfg(module_mask, model_cfg):
    # Note: the first conv layer is a normal conv layer rather than a MaskConvBN layer.
    module_cfg = [model_cfg[0]]
    first_conv_pseudo_mask = torch.ones(model_cfg[0][1], dtype=module_mask.dtype, device=module_mask.device)
    module_layer_masks = [first_conv_pseudo_mask]

    point = 0
    new_in_channels = module_cfg[0][1]
    for i, item in enumerate(model_cfg[1:], start=1):
        _, out_channels, stride, is_depthwise = item
        if i == 1:
            # because of (1) keeping first conv layer intact,
            # and (2) the constraint of group conv (i.e., num channels of two conse. layers are the same)
            # -> the first conv layer in block_1 also need to keep intact
            each_layer_mask = copy.deepcopy(first_conv_pseudo_mask)
        else:
            each_layer_mask = module_mask[point: point + out_channels]
        new_out_channels = torch.sum(each_layer_mask).cpu().int().item()

        if is_depthwise:  # handle group conv
            if i > 1:  # skip modifying (previous) first conv layer
                intersection_mask = each_layer_mask & module_layer_masks[i - 1]
                if intersection_mask.sum() <= 0:
                    # if there is no intersection, create a minimal (1-element) intersection mask
                    intersection_mask = torch.zeros(each_layer_mask.shape, dtype=each_layer_mask.dtype,
                                                    device=each_layer_mask.device)
                    first_true_index = each_layer_mask.argmax(0)
                    intersection_mask[first_true_index] = 1
                module_layer_masks[i - 1] = intersection_mask
                module_cfg[i - 1][1] = torch.sum(module_layer_masks[i - 1]).cpu().int().item()
                new_in_channels = new_out_channels

        module_layer_masks.append(each_layer_mask)
        module_cfg.append([new_in_channels, new_out_channels, stride, is_depthwise])

        new_in_channels = new_out_channels
        point += out_channels

    assert len(module_mask) == point
    return module_cfg, module_layer_masks


def get_module_param(module, module_layer_masks, model_param, keep_generator=True):
    """
    get the module's parameters by removing the irrelevant kernels.
    """
    module_param = copy.deepcopy(model_param)
    module_layer_masks = module_layer_masks[1:]  # remove the pseudo mask of the first conv layer

    masked_conv_idx = 0

    for block_name, each_block in module.named_modules():
        if not block_name.startswith("conv_block"):
            continue
        for each_layer_idx, each_layer in each_block._modules.items():
            if not isinstance(each_layer, MaskConvBN):
                continue
            layer_mask = module_layer_masks[masked_conv_idx]
            retrained_kernel_indices = torch.nonzero(layer_mask, as_tuple=True)[0]

            for layer_param_name in ['conv.weight', 'bn.weight', 'bn.bias',
                                     'bn.running_mean', 'bn.running_var']:
                full_name = f'{block_name}.{each_layer_idx}.{layer_param_name}'
                model_layer_param = model_param[full_name]
                module_layer_param = model_layer_param[retrained_kernel_indices]
                if layer_param_name == 'conv.weight':
                    if masked_conv_idx > 0 and int(each_layer_idx) > 0:  # group convolutions, skip this
                        previous_layer_mask = module_layer_masks[masked_conv_idx - 1]
                        previous_retrained_kernel_indices = torch.nonzero(previous_layer_mask, as_tuple=True)[0]
                        module_layer_param = module_layer_param[:, previous_retrained_kernel_indices, :, :]
                module_param[full_name] = module_layer_param

            masked_conv_idx += 1

    assert masked_conv_idx == len(module_layer_masks)

    # modify the first Linear layer's input dimension
    previous_retrained_kernel_indices = torch.nonzero(module_layer_masks[-1], as_tuple=True)[0]
    layer_param_name = 'fc.weight'
    model_linear_weight = model_param[layer_param_name]
    module_param[layer_param_name] = model_linear_weight[:, previous_retrained_kernel_indices]

    if not keep_generator:
        new_module_param = {}
        for param_name in module_param:
            if 'mask_generator' not in param_name:
                new_module_param[param_name] = module_param[param_name]
        module_param = new_module_param
    return module_param


def mobilenet_module(model_param, module_mask, keep_generator=True, **kwargs):
    module_cfg, module_layer_masks = get_module_cfg(module_mask, DEFAULT_MOBILENET_CONV_CFG)
    module = MobileNet_Module(conv_configs=module_cfg, keep_generator=keep_generator, **kwargs)
    module.build_forward_padding_layer(module_layer_masks)

    module_param = get_module_param(module, module_layer_masks, model_param, keep_generator=keep_generator)
    module.float64_param = module_param
    module.load_state_dict(module_param)
    return module
