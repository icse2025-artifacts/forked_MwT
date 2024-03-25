import argparse
import os

import torch

from configs import Configs
from models_cnnsplitter.rescnn_masked import ResCNN as mt_rescnn_model

from models_cnnsplitter.simcnn_masked import SimCNN as mt_simcnn_model
from models.resnet_masked import ResNet18 as mt_ResNet18_model
from models.vgg_masked import cifar10_vgg16_bn as mt_vgg16_model
from dataset_loader import load_cifar10_target_class, load_svhn_target_class, load_cifar100_target_class
from model_utils import powerset, print_model_summary
from modularizer import evaluate_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vgg16', 'resnet18', 'simcnn', 'rescnn'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn', 'cifar100'], required=True)

    parser.add_argument('--lr_model', type=float, default=0.05)
    # parser.add_argument('--lr_mask', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--batch_size', type=int, default=128)

    # parser.add_argument('--target_classes', nargs='+', type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    if model_name == 'vgg16':
        mt_model = mt_vgg16_model(pretrained=False).to(DEVICE)
    elif model_name == 'resnet18':
        mt_model = mt_ResNet18_model(num_classes=10).to(DEVICE)
    elif model_name == 'simcnn':
        mt_model = mt_simcnn_model().to(DEVICE)
    elif model_name == 'rescnn':
        mt_model = mt_rescnn_model().to(DEVICE)
    else:
        raise ValueError

    print_model_summary(mt_model)

    num_classes = mt_model.num_classes
    all_classes = list(range(num_classes))
    for target_classes in list(powerset(all_classes))[::-1]:
        if len(target_classes) <= 1:
            continue

        if dataset_name == 'cifar10':
            target_train_loader, target_test_loader = load_cifar10_target_class(
                configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, target_classes=target_classes,
                transform_label=False)
        elif dataset_name == 'svhn':
            target_train_loader, target_test_loader = load_svhn_target_class(
                f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=num_workers,
                target_classes=target_classes, transform_label=False)
        elif dataset_name == 'cifar100':
            target_train_loader, target_test_loader = load_cifar100_target_class(
                configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, target_classes=target_classes,
                transform_label=False)
        else:
            raise ValueError

        mt_model.load_state_dict(torch.load(mt_model_save_path, map_location=DEVICE))
        mt_model_acc = evaluate_model(mt_model, target_test_loader)
        print(f"{mt_model_acc:.2f} | {target_classes}")


if __name__ == '__main__':
    args = get_args()
    DEVICE = torch.device('cuda')

    # print(args)
    # print('-' * 100)
    model_name = args.model
    dataset_name = args.dataset
    lr_model = args.lr_model
    lr_mask = lr_model
    # lr_mask = args.lr_mask
    alpha = args.alpha
    beta = args.beta
    batch_size = args.batch_size
    # THRESHOLD = args.threshold
    # target_classes = args.target_classes
    # print(f'TCs: {target_classes}')

    num_workers = 2

    configs = Configs()

    save_dir = f'{configs.data_dir}/{model_name}_{dataset_name}'
    mt_model_save_path = f'{save_dir}/lr_model_mask_{lr_model}_{lr_mask}_a_{alpha}_b_{beta}_bz_{batch_size}.pth'
    print("Evaluating:", mt_model_save_path)

    mt_model_save_dir = os.path.dirname(mt_model_save_path)
    if not os.path.exists(mt_model_save_dir):
        os.makedirs(mt_model_save_dir)

    main()
