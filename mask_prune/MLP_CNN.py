import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d
import numpy as np
# MaskedLinear = nn.Linear
# MaskedConv2d = nn.Conv2d

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28 * 28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = MaskedConv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = MaskedLinear(64 * 8 * 8, 10)

        # 共有多少层（这里只计算conv）
        self.count=0

        self.accm_grad = []

        # 存放计算出来的mask，一个mask组成的list
        self.mask = []

        self.init_some()

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])

    def init_some(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Linear):
                self.count+=1
                self.accm_grad.append(torch.zeros_like(m.weight.data).cuda())
                self.mask.append(torch.ones_like(m.weight.data).cuda())


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

    # #def stb_cluster_gn_mask(self, k=3):
    #
    # def rand_ratio_gn_mask(self, ratio):
    #     count = 0
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):  # or isinstance(m, nn.Linear):
    #             rand_tensor = torch.rand_like(m.weight.data)
    #             abs_tensor = torch.abs(rand_tensor*self.mask[count])
    #             abs_array = abs_tensor.clone().cpu().numpy().flatten()
    #             prun_shreshold = np.percentile(np.array(abs_array), ratio)
    #
    #             tmp_prune_mask = abs_tensor > prun_shreshold
    #
    #             # print(prun_shreshold)
    #             # print(abs_array)
    #
    #             self.mask[count] = tmp_prune_mask.float()
    #
    #             count+=1