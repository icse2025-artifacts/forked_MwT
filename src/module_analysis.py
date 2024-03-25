import argparse
import copy
import itertools
import math
import os
import random

import torch
from torch import nn

from configs import Configs
from models_cnnsplitter.rescnn_masked import ResCNN as mt_rescnn_model

from models_cnnsplitter.simcnn_masked import SimCNN as mt_simcnn_model
from models.resnet import ResNet18 as st_ResNet18_model
from models.resnet_masked import ResNet18 as mt_ResNet18_model

from models.vgg_masked import cifar10_vgg16_bn as mt_vgg16_model
from models.vgg import cifar10_vgg16_bn as st_vgg16_model
from models.mobilenet_masked import MobileNet as mt_MobileNet_model
from models.mobilenet import MobileNet as st_MobileNet_model
from dataset_loader import load_cifar10_target_class, load_svhn_target_class, load_cifar100_target_class
from model_utils import powerset, print_model_summary
from modularizer import evaluate_model, generate_all_modules_masks
from model_utils import count_parameters

from modules_arch.vgg_module_v2 import cifar10_vgg16_bn as vgg16_module
from modules_arch.resnet_module_v2 import ResNet18 as ResNet18_module
from modules_arch.mobilenet_module_v2 import mobilenet_module as MobileNet_module

DEVICE = torch.device('cuda')


def generate_target_module(target_classes, module_mask_path, mt_model_param):
    # load modules' masks and the pretrained model
    all_modules_masks = torch.load(module_mask_path, map_location=DEVICE)
    target_module_mask = (torch.sum(all_modules_masks[target_classes], dim=0) > 0).int()

    num_classes = list(mt_model_param.items())[-1][1].shape[0]
    if model_name == 'vgg16':
        module = vgg16_module(model_param=mt_model_param, module_mask=target_module_mask,
                              keep_generator=False, num_classes=num_classes).to(DEVICE)
    elif model_name == 'resnet18':
        module = ResNet18_module(model_param=mt_model_param, module_mask=target_module_mask,
                                 keep_generator=False, num_classes=num_classes).to(DEVICE)
    elif model_name == 'mobilenet':
        module = MobileNet_module(model_param=mt_model_param, module_mask=target_module_mask,
                                  keep_generator=False, num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError

    return module


def evaluate_modules():
    if model_name == 'vgg16':
        st_model = st_vgg16_model(pretrained=False, num_classes=num_classes).to(DEVICE)
    elif model_name == 'resnet18':
        st_model = st_ResNet18_model(num_classes=num_classes).to(DEVICE)
    elif model_name == 'mobilenet':
        st_model = st_MobileNet_model(num_classes=num_classes).to(DEVICE)

    model_num_classes = num_classes
    model_param_count = count_parameters(st_model)

    if model_name == 'vgg16':
        mt_model = mt_vgg16_model(pretrained=False, num_classes=num_classes).to(DEVICE)
    elif model_name == 'resnet18':
        mt_model = mt_ResNet18_model(num_classes=num_classes).to(DEVICE)
    elif model_name == 'mobilenet':
        mt_model = mt_MobileNet_model(num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError

    mt_model_param = mt_model.state_dict()
    trackable_params = generate_trackable_params(mt_model_param)
    modules = []
    module_total_sizes = []
    for curr_class in range(model_num_classes):
        curr_module = generate_target_module([curr_class], modules_masks_save_path, copy.deepcopy(trackable_params))
        modules.append(curr_module)
        curr_module_param_count = count_parameters(curr_module)
        module_total_sizes.append(curr_module_param_count / model_param_count)
        print(f"[Class {curr_class}] Module's param count: {curr_module_param_count:,} "
              f"({curr_module_param_count / model_param_count:.2f})")
    module_overlap_sizes = calculate_overlap_params(modules, model_param_count)
    # print(model_type)
    # print(raw_model)
    # print(modular_masks_path)
    # print(module_total_sizes)
    # print(module_overlap_sizes)
    print("module_total_size", sum(module_total_sizes) / len(module_total_sizes))
    print("module_overlap_sizes", sum(module_overlap_sizes) / len(module_overlap_sizes))


def generate_trackable_params(raw_model_params):
    # generate unique values to params to measure the overlap after those params are modularized to individual modules
    # this means the pretrained params will be replaced (so don't use it for evaluate accuracy of the model)

    trackable_model_params = {}
    unique_number = 0
    for param_name, params in raw_model_params.items():
        numel = params.numel()  # Total number of elements in the tensor
        new_tensor = torch.arange(unique_number, unique_number + numel, dtype=torch.float64)
        unique_number += numel
        trackable_model_params[param_name] = new_tensor.view(params.shape)

    return trackable_model_params


def calculate_overlap_params(modules, model_param_count):
    flatten_module_params = []
    for i, m in enumerate(modules):
        flatten_param_set = set()
        # for p in m.parameters():
        for p in m.float64_param.values():
            flatten_param_set.update(p.view(-1).tolist())
        flatten_module_params.append(flatten_param_set)

    # Calculate all combinations of modules
    indices_combinations = list(itertools.combinations(range(len(flatten_module_params)), 2))

    # 357 = sampling from population of [4950 indices_combinations] with confidence level of 95%, margin of error 5%
    # indices_combinations = random.sample(indices_combinations, k=357)

    print("indices_combinations", len(indices_combinations))
    overlap_sizes = []
    for module1_index, module2_index in indices_combinations:
        module1_params = flatten_module_params[module1_index]
        module2_params = flatten_module_params[module2_index]
        curr_intersection = len(module1_params & module2_params)
        # curr_union = len(module1_params | module2_params)
        overlap_sizes.append(curr_intersection / model_param_count)
    return overlap_sizes


def main():
    evaluate_modules()
    # if not os.path.exists(modules_masks_save_path):
    #     generate_all_modules_masks()
    #
    # # load the target module
    # module = generate_target_module([0], modules_masks_save_path, torch.load(mt_model_save_path, map_location=DEVICE))
    #
    # module__num_params = count_parameters(module)
    # print_model_summary(module)
    # print(module__num_params)


if __name__ == '__main__':
    # print(args)
    # print('-' * 100)
    # model_name = "vgg16"
    model_name = "resnet18"
    # model_name = "mobilenet"
    # dataset_name, num_classes = "svhn", 10
    dataset_name, num_classes = "cifar10", 10
    # dataset_name, num_classes = "cifar100", 100

    # THRESHOLD = args.threshold
    # target_classes = args.target_classes
    # print(f'TCs: {target_classes}')

    num_workers = 2

    configs = Configs()

    lr_model = lr_mask = 0.05
    batch_size = 128
    THRESHOLD = 0.9

    alpha = 0.5
    beta = 1.5

    # alpha = 1.0
    # beta = 2.0

    # alpha = 1.3
    # beta = 1.5

    save_dir = f'{configs.data_dir}/{model_name}_{dataset_name}'
    modules_masks_save_path = f'{save_dir}/modules/' \
                              f'lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}/mask_thres_{THRESHOLD}.pth'
    mt_model_save_path = f'{save_dir}/lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}.pth'
    print(model_name, dataset_name, modules_masks_save_path)

    main()
