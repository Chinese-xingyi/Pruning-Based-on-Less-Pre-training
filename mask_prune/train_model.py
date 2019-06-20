import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from MLP_CNN import ConvNet


# Hyper Parameters
param = {
    #'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}


# Data loaders
train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset,
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)


# Load the pretrained model
net = ConvNet()


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
print("--- Pretrained network loaded ---")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
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


def train(model, loss_fn, optimizer, param, loader_train, loader_test, loader_val=None):

    model.train()
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()



            optimizer.step()

        test(model, loader_test)

# train and pruning during
def train_pp(model, loss_fn, optimizer, param, loader_train, loader_test, ratio_list, k=3, loader_val=None):
    model.train()
    ratio_count = 0
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()

            model.update_grad()
            optimizer.step()

        test(model, loader_test)

        if epoch%k == 0 and epoch!=0 and ratio_count<len(ratio_list):
            model.procedure_stb_pruning(ratio_list[ratio_count])
            # model.procedure_weight_pruning(ratio_list[ratio_count])

            ratio_count+=1
            print(ratio_count, 'th','###########################pruning once######################################')


## 对网络随机生成一个mask，训练
def train_rand(model, loss_fn, optimizer, param, loader_train, loader_test, ratio, loader_val=None):
    model.train()
    model.rand_mask(ratio)
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        test(model, loader_test)



# ratio_list =[50, 80, 90, 95, 97.5, 99]
ratio_list =[10, 20, 30, 40, 50, 80]
# train_pp(net, criterion, optimizer, param, loader_train, loader_test, ratio_list, 5)
# train_rand(net, criterion, optimizer, param, loader_train, loader_test, 90)

prune_rate(net)

torch.save(net.state_dict(), 'models/cnn_pretrained.pkl')
