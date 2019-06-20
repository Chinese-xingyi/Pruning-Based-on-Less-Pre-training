'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from pruning.layers import MaskedLinear, MaskedConv2d

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(326, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        self.count = 0

        self.accm_grad = []

        # 存放计算出来的mask，一个mask组成的list
        self.mask = []

        self.init_some()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        ind = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.set_mask(masks[ind])
                ind+=1
                print('set mask ok')

        # self.conv1.set_mask(masks[0])
        # self.conv2.set_mask(masks[1])
        # self.conv3.set_mask(masks[2])

    def init_some(self):
        # count=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Linear):
                self.count+=1
                self.accm_grad.append(torch.zeros_like(m.weight.data).cuda())
                self.mask.append(torch.ones_like(m.weight.data).cuda())
                # m.set_mask(torch.ones_like(m.weight.data).cpu())  #.cuda()
                # # print(count); count+=1


    # 执行一次grad累加，具体计算mask的时候需要累加k次
    def update_grad(self):
        count=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Linear):
                self.accm_grad[count]+=m.weight.grad.data.clone()
                count+=1

    def zero_accmgrad(self):
        for count in range(self.count):
            self.accm_grad[count] = torch.zeros_like(self.accm_grad[count])

    # def stb_ratio_gn_mask(self, new_ratio,  k=3):
    #     count=0
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Linear):
    #             # 可用的grad的绝对值，即已经删除的部分参数刨掉了
    #             mean_grad = torch.abs((self.accm_grad[count]*self.mask[count])/k)
    #             mean_grad_array = mean_grad.clone().cpu().numpy().flatten()
    #             prun_shreshold = np.percentile(np.array(mean_grad_array), 1-new_ratio)
    #
    #             tmp_prune_mask = mean_grad < prun_shreshold
    #
    #             self.mask[count] = tmp_prune_mask.float()
    #
    #             count+=1

    # 更新mask,带grad预测潜力的
    def com_mask(self, new_ratio, old_ratio):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # or isinstance(m, nn.Linear):
                # 首先根据weight，filter的norm2，来计算必留、必剪、不确定的mask
                # 求取每个filters的2范数
                abs_weight= torch.abs(m.weight.data.clone() * self.mask[count])
                weight_norm_list = []
                for i in abs_weight:
                    weight_norm_list.append(torch.norm(i, 2,).item())
                weight_norm = torch.from_numpy(np.array(weight_norm_list)).float()

                # 根据上一轮和这一轮的剪枝精度，以及weight，来三支决策是否保留
                #首先找到三支分割比例:prune    prune_ratio  not sure    keep_ratio   keep
                keep_ratio = new_ratio + (new_ratio-old_ratio)/2
                if keep_ratio>100:
                    keep_ratio = 100.0
                prune_ratio = old_ratio + (new_ratio - old_ratio)/2

                print('stability_pruning:keep_ratio, prune_ratio',keep_ratio, prune_ratio, '11111111111111111111111111111')

                # compte shreshold
                weight_norm_coupy1 = weight_norm.clone(); weight_norm_coupy2 = weight_norm.clone()
                ndarray_weight = weight_norm.clone().cpu().numpy().flatten()
                keep_shreshold = np.percentile(np.array(ndarray_weight), keep_ratio)
                prune_shreshold = np.percentile(np.array(ndarray_weight), prune_ratio)

                # 根据weight构造（0，0，1） 和 （0，1，0）的tensor
                #（0，0，1）
                keep_index = (weight_norm>keep_shreshold).float()

                # print((keep_index))
                print('keep,ratio', keep_index.sum(), len(keep_index))
                #(0,1,0)
                # a = (weight_norm>prune_shreshold).float()
                # # print(type(weight_norm[0].item()), type(a[0].item()))
                # b= a*weight_norm.float()
                # c=b<keep_shreshold
                # tmp_index = c.float()
                tmp_index = (((weight_norm>prune_shreshold).float()*weight_norm).float() < keep_shreshold).float()

                print('tmp_index', tmp_index.sum(), len(tmp_index))

                print(tmp_index+keep_index)
                # 根据累计grad，来决定不确定区域的保留剪去,潜力求保留还是剪枝
                # 先求accm_grad 二范数
                grad_norm_list = []
                for i in self.accm_grad[count]*self.mask[count]:
                    grad_norm_list.append(torch.norm(i, 2).item())
                grad_norm = torch.from_numpy(np.array(grad_norm_list)).float()

                # 乘以 tmp_ind,不用判断的部分全0， 需要判别的filter才保留了potenility
                unsure_potential = grad_norm*tmp_index
                # print('unsure_potential', unsure_potential)

                potential_ndarray = unsure_potential.clone().cpu().numpy().flatten()
                unsure_shre = np.percentile(np.array(potential_ndarray), 100-(new_ratio-old_ratio)/2)

                unsure_keep = (unsure_potential> unsure_shre).float()

                print('unsure_keep', unsure_keep.sum(), len(unsure_keep))

                # 这里只是用0.1 tensor 指示出了，对应位置的filter应该保留还是prune； 还需要进一步调节本层mask，以及下一层的mask
                new_mask_ind = keep_index+unsure_keep

                # 设置一个检测是否有问题的程序——是否全是0、1，是否只有0或者只有1
                for i in range(len(new_mask_ind)):
                    if new_mask_ind[i].item() == 0:
                        # 本层tensor,整个filter 修改mask
                        self.mask[count][i] = self.mask[count][i]*0.0
                        #下一层，每个filter中第i个channel应该改为0
                        if count!= self.count-1:
                            for ind in range(self.mask[count+1].size()[0]):
                                self.mask[count+1][ind][i]=self.mask[count+1][ind][i]*0.0


                    elif new_mask_ind[i].item() == 1:
                        pass
                    else:
                        print('new_mask_ind computation is wrong')



                # # 可用的grad的绝对值，即已经删除的部分参数刨掉了
                # mean_grad = torch.abs((self.accm_grad[count] * self.mask[count]) / k)
                # mean_grad_array = mean_grad.clone().cpu().numpy().flatten()
                # prun_shreshold = np.percentile(np.array(mean_grad_array), 1 - new_ratio)
                #
                # tmp_prune_mask = mean_grad < prun_shreshold
                #
                # self.mask[count] = tmp_prune_mask.float()

                count += 1


    #计算新mask，通过纯weight范数
    def com_mask2(self, new_ratio, old_ratio):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # or isinstance(m, nn.Linear):
                # 首先根据weight，filter的norm2，来计算必留、必剪、不确定的mask
                # 求取每个filters的2范数
                abs_weight= torch.abs(m.weight.data.clone() * self.mask[count])
                weight_norm_list = []
                for i in abs_weight:
                    weight_norm_list.append(torch.norm(i, 2,).item())
                weight_norm = torch.from_numpy(np.array(weight_norm_list)).float()

                # 根据上一轮和这一轮的剪枝精度，以及weight，来三支决策是否保留
                #首先找到三支分割比例:prune    prune_ratio  not sure    keep_ratio   keep
                keep_ratio = new_ratio
                if keep_ratio>100:
                    keep_ratio = 100.0
                prune_ratio = old_ratio

                print('stability_pruning:keep_ratio, prune_ratio',keep_ratio, prune_ratio, '11111111111111111111111111111')

                # compte shreshold
                weight_norm_coupy1 = weight_norm.clone(); weight_norm_coupy2 = weight_norm.clone()
                ndarray_weight = weight_norm.clone().cpu().numpy().flatten()
                keep_shreshold = np.percentile(np.array(ndarray_weight), keep_ratio)
                prune_shreshold = np.percentile(np.array(ndarray_weight), prune_ratio)

                # 根据weight构造（0，0，1） 和 （0，1，0）的tensor
                #（0，0，1）
                keep_index = (weight_norm>keep_shreshold).float()

                # 这里只是用0.1 tensor 指示出了，对应位置的filter应该保留还是prune； 还需要进一步调节本层mask，以及下一层的mask
                new_mask_ind = keep_index

                # 设置一个检测是否有问题的程序——是否全是0、1，是否只有0或者只有1
                for i in range(len(new_mask_ind)):
                    if new_mask_ind[i].item() == 0:
                        # 本层tensor,整个filter 修改mask
                        self.mask[count][i] = self.mask[count][i]*0.0
                        #下一层，每个filter中第i个channel应该改为0
                        if count!= self.count-1:
                            for ind in range(self.mask[count+1].size()[0]):
                                self.mask[count+1][ind][i]=self.mask[count+1][ind][i]*0.0


                    elif new_mask_ind[i].item() == 1:
                        pass
                    else:
                        print('new_mask_ind computation is wrong')

                count += 1
                '''
                self.mask[count] = torch.zeros_like(m.weight.data)
                count+=1'''



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
'D_yanzheng': [41, 59, 'M', 82, 118, 'M', 163, 235, 326, 'M', 470, 326, 470, 'M', 326, 470, 326, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))

def modified_vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D_yanzheng'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
