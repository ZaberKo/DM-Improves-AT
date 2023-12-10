import os
import mlconfig
import torch
from .resnet import PreActResNet
from .robnet import RobNet
from .advrush import AdvRush

# Setup mlconfig
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.Adamax)

mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

mlconfig.register(torch.nn.CrossEntropyLoss)

mlconfig.register(PreActResNet)
mlconfig.register(RobNet)
mlconfig.register(AdvRush)

config_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "configs")



def get_robust_resnet_model(dataset, mode='A1'):
    if dataset in ['cifar10', 'cifar10s']:
        config = mlconfig.load(os.path.join(config_path, 'CIFAR10', f'RobustResNet-{mode}.yaml'))
        model = mlconfig.instantiate(config.model)
    elif dataset in ['cifar100', 'cifar100s']:
        config = mlconfig.load(os.path.join(config_path, 'CIFAR100', f'RobustResNet-{mode}.yaml'))
        model = mlconfig.instantiate(config.model)
    else:
        raise ValueError(f'Dataset {dataset} is not supported!')
    
    return model
