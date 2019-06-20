
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from vgg import vgg19_bn
# from vgg_hard_prune import vgg19_bn
from myprune import prune_vgg
import math


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 500,
    'learning_rate': 0.05,
    'weight_decay': 5e-4,
    'momentum':0.9,
}

# Data loaders
# train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True,
#     transform=transforms.ToTensor())
# loader_train = torch.utils.data.DataLoader(train_dataset,
#     batch_size=param['batch_size'], shuffle=True)
#
# test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,
#     transform=transforms.ToTensor())
# loader_test = torch.utils.data.DataLoader(test_dataset,
#     batch_size=param['test_batch_size'], shuffle=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

loader_train =  torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, pin_memory=True)

loader_test = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, pin_memory=True)


# Load the pretrained model
net = vgg19_bn()

# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         m.set_mask(torch.rand((2,3,4)))
#print('ok')

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
print("--- Pretrained network loaded ---")

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
#                                 weight_decay=param['weight_decay'])
optimizer = torch.optim.SGD(net.parameters(), param['learning_rate'],
                                momentum=param['momentum'],
                                weight_decay=param['weight_decay'])


def test(model, loader):
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100. * acc,
        num_correct,
        num_samples,
    ))

    return acc


net =  torch.load('models/vgg_real_pruned80_bestall(2).pkl')

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        print(m.weight.data.size())

# test(net, loader_test)