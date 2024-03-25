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
import torch.nn.functional as F


def calculate_module_losses(layer_activation_values, labels):
    """
    Inherit from https://github.com/qibinhang/MwT/blob/3d29e7eb2c1505c8d3b3e112b8dd5a06fd49fd9b/src/modular_trainer.py#L152
    """

    sum_coupling = torch.tensor(0.0, device=labels.device)
    sum_cohesion = torch.tensor(0.0, device=labels.device)
    sum_compactness = torch.tensor(0.0, device=labels.device)

    same_label_mask = (labels[:, None] == labels[None, :])
    valid_sample_pair_mask = torch.triu(torch.ones_like(same_label_mask,
                                                        device=labels.device,
                                                        dtype=torch.bool), diagonal=1)
    # flattening indices ".view(-1).nonzero().squeeze()"
    # is just for performance optimization of indexing
    # for understanding the rationale behind, just ignore the flattening part
    same_label_indices = (valid_sample_pair_mask & same_label_mask).view(-1).nonzero().squeeze()
    diff_label_indices = (valid_sample_pair_mask & ~same_label_mask).view(-1).nonzero().squeeze()

    for curr_layer_act in layer_activation_values:
        if len(curr_layer_act.shape[1:]) == 3:  # reshape Conv2d layer output
            # consider a feature map as a neuron (by averaging its act values)
            transformed_layer_act = curr_layer_act.abs().mean(dim=(2, 3))
        else:
            transformed_layer_act = curr_layer_act

        norm_acts = F.normalize(transformed_layer_act, p=2, dim=1)
        act_sim = torch.matmul(norm_acts, norm_acts.T).view(-1)

        sum_cohesion += act_sim.index_select(0, same_label_indices).mean()
        sum_coupling += act_sim.index_select(0, diff_label_indices).mean()
        sum_compactness += transformed_layer_act.norm(p=1) / transformed_layer_act.numel()

    num_layers = len(layer_activation_values)
    loss_cohesion = 1 - sum_cohesion / num_layers
    loss_coupling = sum_coupling / num_layers
    loss_compactness = sum_compactness / num_layers
    return loss_cohesion, loss_coupling, loss_compactness


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


def mean_list(input_list):
    if len(input_list) == 0:
        return torch.tensor(0)
    result = sum(input_list) / len(input_list)
    if isinstance(result, torch.Tensor):
        result = result.item()
    return result


@torch.no_grad()
def measure_modular_metrics(model, train_loader):
    model.to(DEVICE)
    model.eval()

    enable_model_tracking_activations(model)

    pbar = tqdm(train_loader, desc="Inference")
    all_loss_overall, all_loss_ce, all_loss_cohesion, \
        all_loss_coupling, all_loss_compactness = [], [], [], [], []

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        modular_activations = [l.modular_activations for l_name, l in get_list_of_layers_need_to_track_act(model)]
        # modular_activations = model.get_masks()
        curr_loss_cohesion, curr_loss_coupling, \
            curr_loss_compactness = calculate_module_losses(modular_activations, labels)

        all_loss_cohesion.append(curr_loss_cohesion.detach())
        all_loss_coupling.append(curr_loss_coupling.detach())
        all_loss_compactness.append(curr_loss_compactness.detach())

    print({"loss_cohesion": mean_list(all_loss_cohesion),
           "loss_coupling": mean_list(all_loss_coupling),
           "loss_compactness": mean_list(all_loss_compactness)})


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
    print_model_summary(model)

    measure_modular_metrics(model, train_loader)


if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    # model_name = "vgg16"
    model_name = "resnet18"
    # model_name = "mobilenet"

    # dataset_name = "svhn"
    dataset_name = "cifar10"
    # dataset_name = "cifar100"

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
