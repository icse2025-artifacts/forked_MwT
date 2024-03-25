import os

import torch
from tqdm import tqdm

from models.vgg_masked import cifar10_vgg16_bn as vgg16
from models.resnet_masked import ResNet18
from models.resnet import ResNet18 as St_Resnet18
from models.mobilenet_masked import MobileNet
from dataset_loader import load_cifar10, load_svhn, load_cifar100
from model_utils import print_model_summary
from configs import Configs
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F


def _hook_to_track_layer_outputs(module, input, output):
    module.modular_activations = output


def get_list_of_layers_need_to_track_act(model):
    return [(l_name, l) for l_name, l in model.named_modules() if
            isinstance(l, nn.ReLU) and "mask_generator" not in l_name]


def enable_model_tracking_activations(model):
    relu_layers = get_list_of_layers_need_to_track_act(model)
    for i, (name, layer) in enumerate(relu_layers):
        # print(i, name, layer)
        layer.register_forward_hook(_hook_to_track_layer_outputs)


@torch.no_grad()
def _get_model_outputs(model, data_loader, device, num_classes=None, target_classes=None, show_progress=True):
    model.to(device)
    model.eval()

    all_outputs = []
    all_labels = []
    pbar = tqdm(data_loader, desc="Collecting Model Outputs", disable=not show_progress)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if target_classes is not None:
            # use for evaluate module acc in case of it classifying only [target classes] instead of [all classes]
            assert num_classes
            if outputs.shape[1] != len(target_classes):
                outputs = outputs[:, target_classes]
            labels = F.one_hot(labels, num_classes=num_classes)[:, target_classes].argmax(dim=1)

        all_outputs.append(outputs)
        all_labels.append(labels)

    return torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)


def activation_extractor(model, train_loader, num_classes):
    nth_last_relu_layer_index = -1

    layer_tuples = get_list_of_layers_need_to_track_act(model)
    nth_last_relu_layer = layer_tuples[nth_last_relu_layer_index]
    nth_last_relu_layer_name = nth_last_relu_layer[0]

    feature_extractor = create_feature_extractor(model, return_nodes={nth_last_relu_layer_name: "out"})
    old_forward_method = feature_extractor.forward
    feature_extractor.forward = lambda x: old_forward_method(x)["out"]

    outputs, labels = _get_model_outputs(feature_extractor, train_loader, device=DEVICE,
                                         num_classes=num_classes,
                                         show_progress=True)
    result = {nth_last_relu_layer_name: outputs.detach().cpu(), "labels": labels.detach().cpu()}
    # with open(os.path.join(".", f"layer_act_values.{model_type}_{dataset_type}.CohesionCouplingCompactness.pt"), "wb") as output_file:
    with open(os.path.join(".", f"./acts/layer_act_values.{model_name}_{dataset_name}.CohesionCoupling.pt"), "wb") as output_file:
    # with open(os.path.join(".", f"layer_act_values.{model_type}_{dataset_type}.STD.pt"), "wb") as output_file:
        torch.save(result, output_file)


def main():
    if dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == 'svhn':
        train_loader, test_loader = load_svhn(f'{configs.dataset_dir}/svhn', batch_size=batch_size,
                                              num_workers=num_workers)
    elif dataset_name == 'cifar100':
        train_loader, test_loader = load_cifar100(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError

    num_classes = len(train_loader.dataset.classes)

    if model_name == 'vgg16':
        model = vgg16(pretrained=False, num_classes=num_classes).to(DEVICE)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes).to(DEVICE)
    elif model_name == 'mobilenet':
        model = MobileNet(num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError

    model.load_state_dict(torch.load(mt_model_save_path, map_location=DEVICE))
    enable_model_tracking_activations(model)
    print_model_summary(model)
    activation_extractor(model, train_loader, num_classes)


if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    model_name = "vgg16"
    # model_name = "resnet18"
    # model_name = "mobilenet"

    # dataset_name = "svhn"
    # dataset_name = "cifar10"
    dataset_name = "cifar100"

    configs = Configs()
    batch_size = 128
    num_workers = 2
    lr_model = lr_mask = 0.05

    alpha = 0.5
    beta = 1.5

    # alpha = 1.0
    # beta = 2.0

    # alpha = 1.3
    # beta = 1.5

    save_dir = f'{configs.data_dir}/{model_name}_{dataset_name}'
    mt_model_save_path = f'{save_dir}/lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}.pth'

    print(model_name, dataset_name)
    main()
