import numpy as np
import torch
from torch.autograd import grad


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_class_gradients(inputs, output):

	assert inputs.requires_grad

	num_classes = output.size()[1]
	grads = []

	for i in range(num_classes):
		dfdx = grad(output[:, i], inputs, grad_outputs=torch.ones(output[:, i].size()).cuda() if output.is_cuda else torch.ones(output[:, i].size()),
					create_graph=True, retain_graph=True)#, only_inputs=True)

		grads.append(dfdx[0])
	return grads  # list of gradients


def epoch(data_loader, model, output_activation=None, train=False, criterion=None, optimizer=None, l1_factor=0.0, clr_factor=0.0):
	if train:
		if not criterion or not optimizer:
			raise AttributeError("criterion and optimizer must be given for training")

	losses = AverageMeter()
	labels = []
	predictions = []

	# switch mode
	if train:
		model.train()
	else:
		model.eval()

	for bi, batch in enumerate(data_loader):

		inputs, targets = batch

		input_var = torch.autograd.Variable(inputs)
		target_var = torch.autograd.Variable(targets)

		if next(model.parameters()).is_cuda:  # returns a boolean
			input_var = input_var.cuda()
			target_var = target_var.cuda()

		# compute output
		output = model(input_var)

		if output_activation:
			output = output_activation(output, dim=1)

		predictions.append(output.data)
		labels.append(targets)

		if l1_factor > 0.0:
			previous_params = []

		if criterion:
			loss = criterion(output, target_var)
			if l1_factor > 0.0:
				l1_crit = torch.nn.L1Loss(size_average=False)
				reg_loss = 0
				for name, param in model.named_parameters():
					if 'input_to_hidden' not in name and 'weight' in name:
						reg_loss += l1_crit(param, torch.zeros_like(param))
						previous_params.append(param.data.clone())
				loss += l1_factor * reg_loss

			# CLR
			if clr_factor > 0.0:

				# compute dummy output
				dummy_input = torch.autograd.Variable(input_var.data.clone(), requires_grad=True)
				dummy_output = model(dummy_input)
				class_gradients = compute_class_gradients(dummy_input, dummy_output)

				# L1 version of CLR
				clr_loss = torch.mean(torch.sum(torch.abs(class_gradients[0] - class_gradients[1]), dim=1))
				loss += clr_factor*clr_loss

			assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

			# measure accuracy and record loss
			losses.update(loss.data[0], inputs.size(0))

		if train:
			# compute gradient and do update step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# L1 Clipping
			if l1_factor > 0.0:
				i = 0
				for name, param in model.named_parameters():
					if 'input_to_hidden' not in name and 'weight' in name:
						param.data.mul_(torch.ge(previous_params[i]*param.data, 0.0).float())
						i += 1

	return torch.cat(labels, 0), torch.cat(predictions, 0), losses.avg
