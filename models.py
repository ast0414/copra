import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
	def __init__(self, input_dim, num_hidden_layers, hidden_dim, dropout=0.5, activation_fn=nn.ReLU):
		super(MLP, self).__init__()
		self.num_hidden_layers = num_hidden_layers

		self.input_to_hidden = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(in_features=input_dim, out_features=hidden_dim),
			activation_fn(),
			nn.BatchNorm1d(num_features=hidden_dim, affine=False)
		)
		init.xavier_normal(self.input_to_hidden[1].weight)
		self.input_to_hidden[1].bias.data.zero_()

		if num_hidden_layers > 1:
			self.hiddens = nn.ModuleList(
				[nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
							   activation_fn(),
							   nn.BatchNorm1d(num_features=hidden_dim, affine=False)
							   ) for i in range(num_hidden_layers - 1)]
			)
			for i in range(num_hidden_layers - 1):
				init.xavier_normal(self.hiddens[i][0].weight)
				self.hiddens[i][0].bias.data.zero_()

		self.output_logit = nn.Linear(in_features=hidden_dim, out_features=2)
		init.xavier_normal(self.output_logit.weight)
		self.output_logit.bias.data.zero_()

	def forward(self, x):
		x = self.input_to_hidden(x)
		if self.num_hidden_layers > 1:
			for hidden in self.hiddens:
				x = hidden(x)
		x = self.output_logit(x)
		return x
