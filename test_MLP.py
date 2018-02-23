import argparse
import numpy as np
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from utils import epoch
from models import MLP

parser = argparse.ArgumentParser()
parser.add_argument('model_path', metavar='MODEL_PATH', help='path to a trained model')
parser.add_argument('csr_path', metavar='CSR_PATH', help='path to feature data stored in a pickled scipy CSR format')
parser.add_argument('label_path', metavar='LABEL_PATH', help='path to true labels stored in a pickled python list')
parser.add_argument('--eval_batch_size', type=int, default=512, help='batch size for eval mode. Default=512')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.set_defaults(cuda=True)


if __name__ == '__main__':
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	with open(args.label_path, 'rb') as f:
		y_test = pickle.load(f)
		y_test = np.array(y_test)

	with open(args.csr_path, 'rb') as f:
		X_test = pickle.load(f)
		X_test = X_test.todense()

	test_set = TensorDataset(torch.from_numpy(X_test.astype('float32')), torch.from_numpy(y_test.astype('int')))
	test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False)

	model = torch.load(args.model_path)
	if args.cuda:
		model = model.cuda()

	_, scores, _ = epoch(test_loader, model, output_activation=F.softmax)

	if args.cuda:
		scores = scores.cpu()
	scores = scores.numpy()
	preds = np.argmax(scores, axis=1)

	y_scores = scores[:, 1]
	y_preds = preds

	auroc = roc_auc_score(y_test, y_scores)
	aupr = average_precision_score(y_test, y_scores)
	f1 = f1_score(y_test, y_preds)
	accuracy = accuracy_score(y_test, y_preds)

	print("AUROC: {}, AUPR: {}, F1: {}, ACC: {}".format(auroc, aupr, f1, accuracy))
