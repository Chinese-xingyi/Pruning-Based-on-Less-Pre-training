
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from MLP_CNN import ConvNet
from vgg import vgg16_bn


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 20,
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
net = vgg16_bn()
new_net = vgg16_bn()

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
    new_net.cuda()
print("--- Pretrained network loaded ---")

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
#                                 weight_decay=param['weight_decay'])
optimizer = torch.optim.SGD(net.parameters(), param['learning_rate'],
                                momentum=param['momentum'],
                                weight_decay=param['weight_decay'])

new_net.load_state_dict(net.state_dict())
count = 0
for m in new_net.modules():
    if isinstance(m, nn.Conv2d):
        new_net.mask[count] = torch.zeros_like(m.weight.data)
        count+=1

new_net.set_masks(new_net.mask)

for parameter in net.parameters():
    print(parameter.data)

for parameter in new_net.parameters():
    print(parameter.data)

prune_rate(new_net)
prune_rate(net)
# # Retraining
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
#                                 weight_decay=param['weight_decay'])
#
# train(net, criterion, optimizer, param, loader_train, loader_test)
#
prune_rate(net)

torch.save(net.state_dict(), 'models/conv_pruned.pkl')
