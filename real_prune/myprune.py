import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_one_conv_layer(model, layer_index, filter_list):
	_, conv = list(model.features._modules.items())[layer_index]
	next_conv = None
	next_con_offset = 1; next_batch_offset=1

	# 从第一层开始提取（而非第0层）
	while layer_index + next_con_offset <  len(model.features._modules.items()):
		res =  list(model.features._modules.items())[layer_index+next_con_offset]
		if isinstance(res[1], torch.nn.modules.conv.Conv2d):
			next_conv_name, next_conv = res
			break
		# 一直在找下一个卷积层
		next_con_offset = next_con_offset + 1

	while layer_index + next_batch_offset <  len(model.features._modules.items()):
		res =  list(model.features._modules.items())[layer_index+next_batch_offset]
		if isinstance(res[1], torch.nn.BatchNorm2d):
			next_batch_name, next_batch = res
			break
		# 一直在找下一个卷积层
		next_batch_offset = next_batch_offset + 1
	
	#print(layer_index, conv)
	# print('!!!!!!')
	# print(next_conv)
	# print(next_conv.out_channels, next_conv.kernel_size, next_conv.stride,next_conv.padding)
	# print('<<<<<')
	new_conv = torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = conv.out_channels - len(filter_list),
			kernel_size = 3, 
			padding = 1)
	
	#new_conv = \
		#torch.nn.Conv2d(in_channels = conv.in_channels, \
			#out_channels = conv.out_channels - 1,
			#kernel_size = conv.kernel_size, \
			#stride = conv.stride,
			#padding = conv.padding)
			#dilation = conv.dilation,
			#groups = conv.groups,
			#bias = True)


	# 将weight和bias转化成 array，再放到新的weight、bias中去
	old_weights = conv.weight.data.cpu().numpy()
	# new_weights = new_conv.weight.data.cpu().numpy()

	new_weights = np.delete(old_weights, filter_list, axis=0)
	# new_weights[: filter_list, :, :, :] = old_weights[: filter_list, :, :, :]
	# new_weights[filter_list : , :, :, :] = old_weights[filter_list + 1 :, :, :, :]
	new_conv.weight.data = torch.from_numpy(new_weights).cuda()

	bias_numpy = conv.bias.data.cpu().numpy()
    #
	# bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
	bias = np.delete(bias_numpy, filter_list, axis=0)
	# bias[:filter_list] = bias_numpy[:filter_list]
	# bias[filter_list : ] = bias_numpy[filter_list + 1 :]
	new_conv.bias.data = torch.from_numpy(bias).cuda()

	# generate new batchnorm layer
	new_batch = torch.nn.BatchNorm2d(conv.out_channels-len(filter_list), eps=next_batch.eps, momentum=next_batch.momentum, affine=next_batch.affine, track_running_stats=next_batch.track_running_stats)
	old_batch_weights = next_batch.weight.data.cpu().numpy()
	old_batch_bias = next_batch.bias.data.cpu().numpy()
	old_batch_running_mean = next_batch.running_mean.data.cpu().numpy()
	old_batch_running_var = next_batch.running_var.data.cpu().numpy()

	new_batch_weights = np.delete(old_batch_weights, filter_list, axis=0)
	new_batch_bias = np.delete(old_batch_bias, filter_list, axis=0)
	new_batch_running_mean = np.delete(old_batch_running_mean, filter_list, axis=0)
	new_batch_running_var = np.delete(old_batch_running_var, filter_list, axis=0)

	new_batch.weight.data.copy_(torch.from_numpy(new_batch_weights).cuda())
	new_batch.bias.data.copy_(torch.from_numpy(new_batch_bias).cuda())
	new_batch.running_mean.data.copy_(torch.from_numpy(new_batch_running_mean).cuda())
	new_batch.running_var.data.copy_(torch.from_numpy(new_batch_running_var).cuda())



	if not next_conv is None:
		next_new_conv = \
		torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = next_conv.out_channels,
			kernel_size = 3, 
			padding = 1)
	
		#new_conv = \
			#torch.nn.Conv2d(in_channels = conv.in_channels, \
				#out_channels = conv.out_channels - 1,
				#kernel_size = conv.kernel_size, \
				#stride = conv.stride,
				#padding = conv.padding)
				#dilation = conv.dilation,
				#groups = conv.groups,
				#bias = True)

		old_weights = next_conv.weight.data.cpu().numpy()
		# new_weights = next_new_conv.weight.data.cpu().numpy()

		new_weights = np.delete(old_weights, filter_list, axis=1)
		# new_weights[:, : filter_list, :, :] = old_weights[:, : filter_list, :, :]
		# new_weights[:, filter_list : , :, :] = old_weights[:, filter_list + 1 :, :, :]
		next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

		next_new_conv.bias.data = next_conv.bias.data

	if not next_conv is None:
	 	features = torch.nn.Sequential(
	            *(replace_layers(model.features, i, [layer_index, layer_index+1, layer_index+next_con_offset], \
	            	[new_conv, new_batch, next_new_conv]) for i, _ in enumerate(model.features)))
	 	del model.features
	 	del conv

	 	model.features = features

	else:
		#Prunning the last conv layer. This affects the first linear layer of the classifier.
	 	model.features = torch.nn.Sequential(
	            *(replace_layers(model.features, i, [layer_index, layer_index+1], \
	            	[new_conv, new_batch]) for i, _ in enumerate(model.features)))

	return model

def prune_vgg(model, layer_list, filters_array):
	for i in range(len(layer_list)):
		if len(filters_array[i])!=0 and len(filters_array[i])-1 == filters_array[i][-1]:
			filters_array[i].pop()
		model = prune_one_conv_layer(model, layer_list[i], filters_array[i])

	return model.cuda()
