基于少量预训练的剪枝：
soft pruning原理，对网络中每个参数都设置一个mask（0或者1组成），即参数tensor和mask的tensor，两个tensor尺寸相同。在网络前向计算时，参数先和mask相乘，再作卷积操作。
             剪去某个参数，该参数对应mask位置设置为0，剪去一个filter，该filter所对应的诸多mask位设置为0。

soft_prune.py:
        重要函数
        gen_mask：在训练过程中，每个少量训练epochs之后生成剪枝参数的mask。其中调用了那两个重要函数：计算mask的函数和添加新mask到网络中的函数
        model.com_mask2(new_ratio,old_ratio)：根据剪枝比例，计算模型的剪枝结果的mask。全网比较filters重要性。输入是上一轮的剪枝比例和本轮的剪枝比例
        model.set_masks(model.mask)：调用即可。和com_mask2()都是模型本身功能

vgg.py, MLP_CNN： 模型文件


test.py： 测试训练好的模型的精度


