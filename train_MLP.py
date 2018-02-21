import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import os
import argparse
import sys
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import epoch
from models import MLP

parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='DATA_PATH', help='path to datasets')
parser.add_argument('--output_dir', type=str, default='./', help='output directory. Default=Current folder')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs. Default=200')
parser.add_argument('--batch_size', type=int, default=512, help='batch size. Default=512')
parser.add_argument('--eval_batch_size', type=int, default=512, help='batch size for eval mode. Default=512')
parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate. Default=1e-1')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum. Default=0.9')
parser.add_argument('--layers', type=int, default=5, help='number of hidden layers. Default=5')
parser.add_argument('--units', type=int, default=256, help='number of hidden units in each layer. Default=256')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate at input layer. Default=0.5')
parser.add_argument('--l1', type=float, default=0.0, help='L1 regularization. Default=0')
parser.add_argument('--clr', type=float, default=0.0, help='Cross-Lipschitz regularization. Default=0')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.set_defaults(cuda=True)


def train_mlp(args, loaders, model, criterion, optimizer, scheduler, l1_factor=0.0, clr_factor=0.0, model_name='model'):
	train_loader = loaders['train_loader']
	valid_loader = loaders['valid_loader']
	test_loader = loaders['test_loader']

	if args.cuda:
		model = model.cuda()

	best_valid_loss = sys.float_info.max

	train_losses = []
	valid_losses = []

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	for i_epoch in tqdm(range(args.epochs), desc='Epochs'):

		# Train
		train_labels, train_preds, train_loss = epoch(train_loader, model, train=True, criterion=criterion, optimizer=optimizer, l1_factor=l1_factor, clr_factor=clr_factor)
		train_losses.append(train_loss)

		# Validation
		valid_labels, valid_preds, valid_loss = epoch(valid_loader, model, criterion=criterion)

		# Learning rate decay
		if scheduler is not None:
			scheduler.step(valid_loss)

		valid_losses.append(valid_loss)

		# remember best valid loss and save checkpoint
		is_best = valid_loss < best_valid_loss

		if is_best:
			best_valid_loss = valid_loss

			# evaluate on test set
			test_labels, test_preds, test_loss = epoch(test_loader, model, criterion=criterion)

			with open(args.output_dir + model_name + '_result.txt', 'w') as f:
				f.write('Best Validation Epoch: {}\n'.format(i_epoch))
				f.write('Best Validation Loss: {}\n'.format(best_valid_loss))
				f.write('Train Loss: {}\n'.format(train_loss))
				f.write('Test Loss: {}\n'.format(test_loss))

			# Save entire model
			torch.save(model, args.output_dir + model_name + '.pth')
			# Save model params
			torch.save(model.state_dict(), args.output_dir + model_name + '_params.pth')

	# plot
	plt.figure()
	plt.plot(np.arange(len(train_losses)), np.array(train_losses), label='Training Loss')
	plt.plot(np.arange(len(valid_losses)), np.array(valid_losses), label='Validation Loss')
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.legend(loc="best")
	plt.savefig(args.output_dir + model_name + '_loss.eps', format='eps')


def load_data(data_path):
	with open(data_path + 'train.data_csr', 'rb') as f:
		X_train = pickle.load(f)
	with open(data_path + 'train.labels', 'rb') as f:
		y_train = pickle.load(f)
	with open(data_path + 'valid.data_csr', 'rb') as f:
		X_valid = pickle.load(f)
	with open(data_path + 'valid.labels', 'rb') as f:
		y_valid = pickle.load(f)
	with open(data_path + 'test.data_csr', 'rb') as f:
		X_test = pickle.load(f)
	with open(data_path + 'test.labels', 'rb') as f:
		y_test = pickle.load(f)

	print('Train: {}, Validation: {}, Test: {}'.format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

	train_set = TensorDataset(torch.from_numpy(X_train.todense().astype('float32')),
							  torch.from_numpy(np.array(y_train).astype('int')))
	valid_set = TensorDataset(torch.from_numpy(X_valid.todense().astype('float32')),
							  torch.from_numpy(np.array(y_valid).astype('int')))
	test_set = TensorDataset(torch.from_numpy(X_test.todense().astype('float32')),
							 torch.from_numpy(np.array(y_test).astype('int')))

	return train_set, valid_set, test_set


if __name__ == '__main__':
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	train_set, valid_set, test_set = load_data(args.data_path)

	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(dataset=valid_set, batch_size=args.eval_batch_size, shuffle=False)
	test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False)
	loaders = {'train_loader': train_loader, 'valid_loader': valid_loader, 'test_loader': test_loader}

	input_dim = train_set.data_tensor.size()[1]

	weight_class0 = torch.mean(train_set.target_tensor.float())
	weight_class1 = 1.0 - weight_class0
	weight = torch.FloatTensor([weight_class0, weight_class1])

	criterion = nn.CrossEntropyLoss(weight=weight)
	if args.cuda:
		criterion = criterion.cuda()

	model = MLP(input_dim=input_dim, num_hidden_layers=args.layers, hidden_dim=args.units, dropout=args.dropout)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	scheduler = ReduceLROnPlateau(optimizer, 'min')

	train_mlp(args, loaders, model, criterion, optimizer, scheduler, l1_factor=args.l1, clr_factor=args.clr)
