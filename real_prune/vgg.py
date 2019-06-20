'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from pruning.layers import MaskedConv2d

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
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 10),
        # )

        self.count = 0

        self.layer_list=[]
        self.filters_array=[]
        self.mask=[]

        self.init_some()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


    def init_some(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.count+=1

        for _,m in list(self.features._modules.items()):
            if isinstance(m, nn.Conv2d):
                # if torch.cuda.is_available():
                #     self.mask.append(torch.ones_like(m.weight.data).cuda())
                # else:
                self.mask.append(torch.ones_like(m.weight.data))

        self.set_masks(self.mask)


    def set_masks(self, masks):
        count = 0
        for _,m in list(self.features._modules.items()):
            if isinstance(m, nn.Conv2d):
                m.set_mask(masks[count])
                count += 1



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
        layer_list = []; filters_array=[]
        count = 0
        count_conv = 0
        for _,m in list(self.features._modules.items()):
            if isinstance(m, nn.Conv2d) and count_conv!=self.count-1:  # or isinstance(m, nn.Linear):
                # 给layer_list\filters_array 添加元素
                layer_list.append(count); filters_array.append([])

                # 首先根据weight，filter的norm2，来计算必留、必剪、不确定的mask
                # 求取每个filters的2范数
                abs_weight= torch.abs(m.weight.data.clone())
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
                ndarray_weight = weight_norm.clone().cpu().numpy().flatten()
                keep_shreshold = float(np.percentile(np.array(ndarray_weight), keep_ratio))
                prune_shreshold = np.percentile(np.array(ndarray_weight), prune_ratio)

                # 根据weight构造（0，0，1） 和 （0，1，0）的tensor
                #（0，0，1）
                keep_index = (weight_norm>keep_shreshold).float()

                # 这里只是用0.1  一维tensor 指示出了，对应位置的filter应该保留还是prune； 还需要进一步调节本层mask，以及下一层的mask
                new_mask_ind = keep_index

                # 设置一个检测是否有问题的程序——是否全是0、1，是否只有0或者只有1
                for i in range(len(new_mask_ind)):
                    if new_mask_ind[i].item() == 0:
                        filters_array[count_conv].append(i)

                    elif new_mask_ind[i].item() == 1:
                        pass
                    else:
                        print('new_mask_ind computation is wrong')

                count_conv+=1
            count += 1
        self.layer_list=layer_list; self.filters_array=filters_array

    #计算新mask，通过纯weight范数,全局计算最小，而com_mask2是每层暴力剪
    def com_mask3(self, new_ratio, old_ratio):
        layer_list = [];filters_array = []
        count = 0;count_conv = 0
        all_norm = []; all_norm_list = []
        for _,m in list(self.features._modules.items()):
            if isinstance(m, nn.Conv2d) and count_conv < self.count - 1:
                # 首先根据weight，filter的norm2，来计算必留、必剪、不确定的mask
                # 求取每个filters的2范数

                ##########################################################################################
                ##########################################################################################
                ##########################################################################################
                # 这里有个大问题，之前有mask'时候，我计算新mask的时候，是排除掉，已剪去的部分的，这里代码，是所有参数重新竞争。问题应该不大？
                ##########3
                ##########################################################################################
                ##########################################################################################
                ##########################################################################################

                abs_weight= torch.abs(m.weight.data.clone())
                weight_norm_list = []
                for i in abs_weight:
                    weight_norm_list.append(torch.norm(i, 2).item())
                #weight_norm = torch.from_numpy(np.array(weight_norm_list)).float()

                # 将每层的weight_norm，放入all_norm中去
                all_norm_list += weight_norm_list
                all_norm.append(weight_norm_list)

                # 添加层号、层filters的[], count对应的是层号,filter序号晚点存入
                layer_list.append(count); filters_array.append([])

                count_conv+=1
            count+=1

        all_layers_shre = float(np.percentile(np.array(all_norm_list), new_ratio))
        print('all_norm[0],length',len(all_norm[0]))
        for i in range(len(all_norm)):
            for j in range(len(all_norm[i])):
                if all_norm[i][j]>=all_layers_shre:
                    all_norm[i][j] = 1
                else:
                    all_norm[i][j] = 0
                    filters_array[i].append(j)

        self.layer_list=layer_list; self.filters_array=filters_array

        all_norm_ind = all_norm

        # 整个网络从第0个卷积层开始，逐层看all_norm_ind的数字情况，确定mask是否调整
        # i是层数，j是filter序号
        i = 0
        for _,m in list(self.features._modules.items()):
            if isinstance(m, nn.Conv2d) and i < self.count - 1:
                # print('i',i)
                # for j in range(len(all_norm_ind[i])):  # 有错!第一层mask，0.1,2,3；然后mask越界了
                count_filer = 0
                for j in range(len(self.mask[i])):
                    # print( )
                    # print('i', i)
                    # print('j', j)
                    if all_norm_ind[i][j]== 0:  #.item() == 0:
                        if count_filer<len(self.mask[i])-2:
                            self.mask[i][j] = self.mask[i][j] * 0.0
                            # 下一层，每个filter中第i个channel应该改为0
                            if i != self.count - 1:
                                for ind in range(self.mask[i + 1].size()[0]):
                                    self.mask[i + 1][ind][j] = self.mask[i + 1][ind][j] * 0.0
                        else:
                            pass

                    elif all_norm_ind[i][j] == 1:  # .item() == 1:
                        pass
                    else:
                        print('new_mask_ind computation is wrong')
                i += 1
            # ######### 强调下，本函数只是用来，更新mask，需要结合set_masks函数S使用


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
    # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
    #       512, 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 10 , 'M'],
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


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))


# ind = 0; ind_conv=0
# print('???')
# model = vgg19_bn()
# print(model.count)
# for _, m in list(model.features._modules.items()):
#     if isinstance(m, nn.Conv2d): #and ind_conv < model.count - 1:
#         print(ind_conv,'?'); print(ind); print('.')
#         ind_conv+=1
#     ind+=1
#
#
# def seva(model, road):
#     torch.save(model.state_dict(), road)
#
#
# r='models/vgg_test333.pkl'
# seva(model, r)


