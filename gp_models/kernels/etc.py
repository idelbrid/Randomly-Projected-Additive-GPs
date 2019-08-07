from torch import nn
import torch
from torch.nn import functional as F


def _sample_from_range(num_samples, range_):
    return torch.rand(num_samples) * (range_[1] - range_[0]) + range_[0]


class DNN(torch.nn.Module):
    """Note: linear output"""
    def __init__(self, input_dim, output_dim, hidden_layer_sizes, nonlinearity='relu', output_activation='linear',
                 separate_networks=False):
            super(DNN, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_layer_sizes = hidden_layer_sizes
            self.nonlinearity = nonlinearity
            self.output_activation = output_activation
            self.separate_networks = separate_networks
            if nonlinearity == 'sigmoid':
                self._nonlinearity = F.sigmoid
            elif nonlinearity == 'relu':
                self._nonlinearity = F.relu_
            elif nonlinearity == 'tanh':
                self._nonlinearity = F.tanh
            elif nonlinearity == 'leakyrelu':
                self._nonlinearity = F.leaky_relu_
            else:
                raise ValueError("Unknown nonlinearity")
            if output_activation == 'linear':
                self._output_activation = lambda x: x
            elif output_activation == 'sigmoid':
                self._output_activation = F.sigmoid
            elif output_activation == 'relu':
                self._output_activation = F.relu
            elif output_activation == 'tanh':
                self._output_activation = F.tanh
            else:
                raise ValueError("Unknown output nonlinearity")

            linear_modules = []
            module_list_list = []
            if not separate_networks:
                layer_sizes = [input_dim]+hidden_layer_sizes + [output_dim]
            else:
                layer_sizes = [input_dim]+hidden_layer_sizes + [1]
            for i in range(1, len(layer_sizes)):
                if not separate_networks:
                    linear_modules.append(
                        nn.Linear(layer_sizes[i-1], layer_sizes[i], bias=False)
                    )
                else:
                    toappend = []
                    for j in range(output_dim):
                        toappend.append(nn.Linear(layer_sizes[i-1], layer_sizes[i], bias=False))
                        linear_modules.append(toappend[-1])
                    module_list_list.append(toappend)

            self.layers = torch.nn.ModuleList(linear_modules)
            self._layer_list = module_list_list

            if not separate_networks:
                for layer in self.layers[:-1]:
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity=nonlinearity)
                nn.init.kaiming_uniform_(self.layers[-1].weight.data, nonlinearity=self.output_activation)
            else:
                for layer_set in self._layer_list[:-1]:
                    for layer in layer_set:
                        nn.init.kaiming_uniform_(layer.weight.data, nonlinearity=nonlinearity)
                for layer in self._layer_list[-1]:
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity=self.output_activation)


    def forward(self, x):
        if not self.separate_networks:
            for i in range(len(self.layers)-1):
                x = self._nonlinearity(self.layers[i](x))
            x = self.layers[-1](x)  # final layer is generally linear
        else:
            output = []
            for j in range(self.output_dim):
                cur_vals = x
                for i in range(len(self._layer_list)-1):
#                     print(cur_vals.shape)
                    cur_vals = self._nonlinearity(self._layer_list[i][j](cur_vals))
#                 print(cur_vals.shape)
                output.append(self._layer_list[-1][j](cur_vals))
            x = torch.cat(output, dim=-1)
#             print(x.shape)

        x = self._output_activation(x)
        return x