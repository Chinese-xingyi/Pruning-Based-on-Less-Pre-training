
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from MLP_CNN import ConvNet
from vgg import modified_vgg16_bn


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 300,
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
net = modified_vgg16_bn()

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




def gen_mask(model, loss_fn, optimizer, param, loader_train, loader_test, ratio, k=3, loader_val=None):
    test(model, loader_test)

    model.train()
    count = 0;
    ratio_ind = 0
    for epoch in range(param['num_epochs']):
        model.train()

        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()

            model.update_grad()

            optimizer.step()
            # print(epoch,t)
            # test(model, loader_test)



        if (epoch + 1) % k == 0 and ratio_ind<len(ratio):
            print(' pruning some filters which are in convolution layes , pruning ratio:%.3f' % ratio[ratio_ind])
            if ratio_ind==0:
                model.com_mask2(ratio[ratio_ind],0)
            else:
                model.com_mask2(ratio[ratio_ind], ratio[ratio_ind-1])
            model.set_masks(model.mask)

            model.zero_accmgrad()
            ratio_ind+=1

        else:
            model.set_masks(model.mask)
        prune_rate(model)


        print('modify learning rate')
        lr = param['learning_rate'] * (0.5 ** ((epoch - k * len(ratio)) // 30))
        # lr = param['learning_rate'] * (0.5 ** ((epoch - 1) // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        print('epoch',epoch)
        test(model, loader_test)
        count+=1


    #model.rand_ratio_gn_mask(ratio)




import time
start = time.time()




# ratio_list =[10.0, 20.0, 30.0, 35.5]
# ratio_list =[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
# ratio_list =[10.0, 20.0,30.0, 40.0,  50.0, 60.0]
ratio_list =[0.0]

# ratio_list = [80.0, 90.0]
gen_mask(net, criterion, optimizer, param, loader_train, loader_test,  ratio_list, 5)

# print('............')
#
#
# # Retraining
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
#                                 weight_decay=param['weight_decay'])
#
# train(net, criterion, optimizer, param, loader_train, loader_test)
#

end = time.time()

print(end-start)

prune_rate(net)

torch.save(net.state_dict(), 'models/conv_pruned_60%.pkl')
