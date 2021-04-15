import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GraphNet import GraphNet
from utils import expand_var_list

class Decoder(nn.Module):

    def __init__(self, nums_nodes, node_sizes, nums_hidden_node_layers, hidden_edge_sizes, output_edge_sizes,
                 nums_mps, dropout, alpha, intensities, batch_norm=True):
        super(Decoder, self).__init__()

        var_lists = [nums_nodes, node_sizes, nums_hidden_node_layers, hidden_edge_sizes, output_edge_sizes, nums_mps, intensities]
        var_types = [type(var) for var in var_lists]
        self.num_layers = 0
        if list in var_types:
            self.num_layers = max(len(var) for var in var_lists if isinstance(var, list))
            self.nums_nodes = expand_var_list(nums_nodes, self.num_layers)
            self.node_sizes = expand_var_list(node_sizes, self.num_layers)
            self.nums_hidden_node_layers = expand_var_list(nums_hidden_node_layers, self.num_layers)
            self.hidden_edge_sizes = expand_var_list(hidden_edge_sizes, self.num_layers)
            self.output_edge_sizes = expand_var_list(output_edge_sizes, self.num_layers)
            self.nums_mps = expand_var_list(nums_mps, self.num_layers)
            self.intensities = expand_var_list(intensities, self.num_layers)
        else:
            raise TypeError("At least one in [nums_nodes, node_sizes, nums_hidden_node_layers, hidden_edge_sizes, output_edge_sizes, nums_mps, intensities] needs to be a list.")

        self.decode = nn.ModuleList()
        for i in range(self.num_layers-1):
            graphNet = GraphNet(num_nodes=self.nums_nodes[i], input_node_size=self.node_sizes[i], output_node_size=self.node_sizes[i+1],
                                num_hidden_node_layers=nums_hidden_node_layers[i], hidden_edge_size=hidden_edge_sizes[i],
                                output_edge_size=output_edge_sizes[i], num_mp=nums_mps[i], dropout=dropout, alpha=alpha,
                                intensity=self.intensities[i], batch_norm=batch_norm)
            self.decode.append(graphNet)

        self.linear = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.linear.append(nn.Linear(self.nums_nodes[i], self.nums_nodes[i+1]))

    def forward(self, x):
        for i in range(self.num_layers-2):
            x = self.encode[i](x)
            x = self.linear[i](x.permute(0,2,1))  # Downsize the node
            x = x.permute(0,2,1)  # Permute it back
        x = self.encode[-1](x)
        return x
