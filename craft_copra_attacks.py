import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import DataLoader, TensorDataset

import os
import argparse
import pickle
from tqdm import tqdm

from scipy.sparse import csr_matrix

from models import MLP

parser = argparse.ArgumentParser()
parser.add_argument('model_path', metavar='MODEL_PATH', help='path to the source model that will be used to craft examples')
parser.add_argument('csr_path', metavar='CSR_PATH', help='path to feature data stored in a pickled scipy CSR format')
parser.add_argument('label_path', metavar='LABEL_PATH', help='path to true labels stored in a pickled python list')
parser.add_argument('--output_dir', type=str, default='./', help='output directory. Default=Current folder')
parser.add_argument('--max-dist', type=int, default=20, help='maximum distortion. Default=20')
parser.add_argument('--early-stop', dest='early_stop', action='store_true', help='Stop perturbing once the label is changed. Default=False')
parser.add_argument('--uncon', dest='constrained', action='store_false', help='craft unconstrained attacks. Default=False')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda. Default=False')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.set_defaults(cuda=True, constrained=True, early_stop=False)


def compute_jacobian(inputs, output):
	"""
	:param inputs: model inputs -> Variable
	:param output: model outputs -> Variable
	:return: Jacobian matrix -> Tensor (num_classes, num_samples, num_features)
	"""
	#from torch.autograd.gradcheck import zero_gradients

	assert inputs.requires_grad

	num_classes = output.size()[1]

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*output.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[:, i] = 1
		output.backward(grad_output, retain_variables=True)
		jacobian[i] = inputs.grad.data

	return jacobian


def saliency_map(jacobian, search_space, target_index, increasing=True):

	all_sum = torch.sum(jacobian, 0).squeeze()
	alpha = jacobian[target_index].squeeze()
	beta = all_sum - alpha

	if increasing:
		mask1 = torch.ge(alpha, 0.0)
		mask2 = torch.le(beta, 0.0)
	else:
		mask1 = torch.le(alpha, 0.0)
		mask2 = torch.ge(beta, 0.0)

	mask = torch.mul(torch.mul(mask1, mask2), search_space)

	if increasing:
		saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
	else:
		saliency_map = torch.mul(torch.mul(torch.abs(alpha), beta), mask.float())

	max_value, max_idx = torch.max(saliency_map, dim=0)

	return max_value, max_idx


# TODO: Assuming that one sample at each time
def copra(model, input_tensor, target_class, max_distortion=10, constrained=True, early_stop=False):
	# max_distortion = int : maximum code deviations (total number of increase/descrease)

	# Make a clone since we will alter the values
	input_features = torch.autograd.Variable(input_tensor.clone(), requires_grad=True)
	num_features = input_features.size(1)
	max_iter = max_distortion  # math.floor(num_features * max_distortion) # Modifying 1 features at each iteration
	count = 0

	# a mask whose values are one for feature dimensions in search space
	search_space = torch.ones(num_features).byte()
	if input_features.is_cuda:
		search_space = search_space.cuda()

	output = model(input_features)
	_, source_class = torch.max(output.data, 1)

	while (count < max_iter) and (search_space.sum() != 0):
		# Calculate Jacobian
		jacobian = compute_jacobian(input_features, output)

		# Restrict changes from 0 to 1 (value should be greater than 0, e.g. 1,2,...)
		if constrained:
			constraint_0_to_1 = torch.ge(input_features.data.squeeze(), 1.0)
			search_space_increasing = torch.mul(constraint_0_to_1, search_space)
		else:
			search_space_increasing = search_space
		increasing_saliency_value, increasing_feature_index = saliency_map(jacobian, search_space_increasing,
																		   target_class, increasing=True)

		# Restrict changes from 1 to 0 (value should be greater than 1, e.g. 2,3,...)
		if constrained:
			constraint_1_to_0 = torch.ge(input_features.data.squeeze(), 2.0)
			search_space_decreasing = torch.mul(constraint_1_to_0, search_space)
		else:
			constraint_negative = torch.ge(input_features.data.squeeze(), 1.0)
			search_space_decreasing = torch.mul(constraint_negative, search_space)
		decreasing_saliency_value, decreasing_feature_index = saliency_map(jacobian, search_space_decreasing,
																		   target_class, increasing=False)

		if increasing_saliency_value[0] == 0.0 and decreasing_saliency_value[0] == 0.0:
			break

		if increasing_saliency_value[0] > decreasing_saliency_value[0]:
			input_features.data[0][increasing_feature_index] += 1
		else:
			input_features.data[0][decreasing_feature_index] -= 1

		output = model(input_features)
		_, source_class = torch.max(output.data, 1)

		count += 1

		if early_stop and (source_class[0] == target_class[0]):
			break

	return input_features


def craft_adv_samples(data_loader, model, max_dist=20, constrained=True, early_stop=False):

	clean_samples = []
	adv_samples = []

	# switch to evaluation mode
	model.eval()

	if constrained:
		print("Constrained Distortion {}".format(max_dist))
	else:
		print("Unonstrained Distortion {}".format(max_dist))

	for bi, batch in enumerate(tqdm(data_loader, desc="Crafting")):

		inputs, targets = batch

		if args.cuda:
			inputs = inputs.cuda()
			targets = targets.cuda()

		# Assuming binary classification
		target_class = 1 - targets

		crafted = copra(model, inputs, target_class, max_distortion=max_dist, constrained=constrained, early_stop=early_stop)
		crafted_adv_samples = crafted.data

		clean_samples.append(inputs)
		adv_samples.append(crafted_adv_samples)

	return torch.cat(clean_samples, 0), torch.cat(adv_samples, 0)


if __name__ == '__main__':
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	with open(args.csr_path, 'rb') as f:
		X_test = pickle.load(f)
	with open(args.label_path, 'rb') as f:
		y_test = pickle.load(f)

	clean_set = TensorDataset(torch.from_numpy(X_test.todense().astype('float32')),
							  torch.from_numpy(np.array(y_test).astype('int')))
	clean_loader = DataLoader(dataset=clean_set, batch_size=1, shuffle=False)

	source_model = torch.load(args.model_path)
	if args.cuda:
		source_model = source_model.cuda()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	clean_samples, adv_samples = craft_adv_samples(clean_loader, source_model, max_dist=args.max_dist, constrained=args.constrained, early_stop=args.early_stop)

	if args.cuda:
		adv_samples = adv_samples.cpu()

	adv_samples = adv_samples.numpy()
	adv_samples = csr_matrix(adv_samples)

	with open(args.output_dir + 'adv_samples.data_csr', 'wb') as f:
		pickle.dump(adv_samples, f, pickle.HIGHEST_PROTOCOL)
