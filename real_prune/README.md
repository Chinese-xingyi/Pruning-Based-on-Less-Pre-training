vgg_real_prune.py： 剪枝vgg
        几个函数：
        gen_mask：基于少量的预训练，判定出哪些filters需要被删除，返回一个剪枝后，真实的“瘦”的网络

        prune_vgg：生成一个瘦的网络。输入：待剪枝网络、需要剪去的filters的层号（list）、该层具体第几个filters（list的list）

        vgg16_bn：待训练的网络（从vgg_hard_prune_morethan70.py中引入）


vgg.py：vgg模型
        几个函数：
        com_mask2：根据vgg网络各层的各个filters 的范数值，确定哪些filters该被剪去。输出的是各层要被剪去的filters的序号



myprune.py 构建剪枝后模型
        几个函数：
        prune_vgg(model, model.layer_list, model.filters_array)
        prune_one_conv_layer（）：针对某卷积层，进行filters剪枝。输入模型、卷积层号、该层需要剪去的filters序号
                                    输出一个某层被剪枝过后的网路