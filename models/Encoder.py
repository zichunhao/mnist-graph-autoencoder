import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GraphNet import GraphNet

class Encoder(nn.Module):

    def __init__(self, num_node, node_sizes, output_node_size, nums_hidden_node_layers, hidden_edge_sizes, output_edge_sizes,
                 nums_mps, dropout, alpha, intensities, batch_norm=True):
        super(Encoder, self).__init__()

        assert len(node_sizes) >= len(nums_hidden_node_layers), f"node_sizes does not have enough elements: (len(node_sizes) ({len(node_sizes)}) < len(nums_hidden_node_layers) ({len(nums_hidden_node_layers)})!"
        assert len(node_sizes) >= len(hidden_edge_sizes), f"node_sizes does not have enough elements: len(node_sizes) ({len(node_sizes)}) < len(hidden_edge_sizes ({len(hidden_edge_sizes)})!"
        assert len(node_sizes) >= len(output_edge_sizes), f"node_sizes does not have enough elements: len(node_sizes) ({len(node_sizes)}) < len(output_edge_sizes) ({len(output_edge_sizes)})!"
        assert len(node_sizes) >= len(nums_mps), f"node_sizes does not have enough elements: len(node_sizes) ({len(node_sizes)}) < len(nums_mps) ({len(nums_mps)})!"
        assert len(node_sizes) >= len(intensities), f"node_sizes does not have enough elements: len(node_sizes) ({len(node_sizes)}) < len(intensities) ({len(intensities)})!"

        for var_list in [nums_hidden_node_layers, hidden_edge_sizes, output_edge_sizes, nums_mps, intensities]:
            expand_var_list(var_list, len(node_sizes))

        self.num_node = num_node
        self.node_sizes = node_sizes
        self.output_node_size = output_node_size
        self.nums_hidden_node_layers = nums_hidden_node_layers
        self.hidden_edge_sizes = hidden_edge_sizes
        self.output_edge_sizes = output_edge_sizes
        self.nums_mps = nums_mps

        self.intensities = intensities

        self.num_layers = len(node_sizes)

        self.encode = nn.ModuleList()
        for i in range(self.num_layers-1):
            graphNet = GraphNet(num_nodes=self.num_node, input_node_size=self.node_sizes[i], output_node_size=self.node_sizes[i+1],
                                num_hidden_node_layers=nums_hidden_node_layers[i], hidden_edge_size=hidden_edge_sizes[i],
                                output_edge_size=output_edge_sizes[i], num_mp=nums_mps[i], dropout=dropout, alpha=alpha,
                                intensity=self.intensities, batch_norm=batch_norm)
            self.encode.append(graphNet)

        graphNet = GraphNet(num_nodes=self.num_node, input_node_size=self.node_sizes[-1], output_node_size=self.output_node_size,
                            num_hidden_node_layers=nums_hidden_node_layers[-1], hidden_edge_size=hidden_edge_sizes[-1],
                            output_edge_size=output_edge_sizes[-1], num_mp=nums_mps[-1], dropout=dropout, alpha=alpha,
                            intensity=self.intensities[-1], batch_norm=batch_norm)
        self.encode.append(graphNet)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.encode[i](x)
        return x

def expand_var_list(lst, length):
    if len(lst) < length:
        lst += [lst[-1]] * length
