import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNet(nn.Module):
    """
    The basic graph neural network in the autoencoder.

    Parameters
    ----------
    num_nodes: int
        The number of nodes for the graph.
    input_node_size: int
        The dimension of input node feature vectors.
    output_node_size: int
        The dimension of output node feature vectors.
    num_hidden_node_layers: int
        The number of layers of hidden nodes.
    hidden_edge_size: int
        The dimension of hidden edges before message passing.
    output_edge_size: int
        The dimenson of output edges for message passing.
    num_mp: int
        The number of message passing step.
    dropout: float
        The dropout value for edge features.
    alpha: float
        The alpha value for the leaky relu layer for edge features.
    intensity: bool
        Whether the lasts element along the last axis are intensities (bool).
        Example: [[x1, x2, I1], ...] and [[x11, x12, x13, ..., I1], ...]
    batch_norm: bool (default: True)
        Whether to use batch normalization.
    """

    def __init__(self, num_nodes, input_node_size, output_node_size, num_hidden_node_layers,
                 hidden_edge_size, output_edge_size, num_mp, dropout, alpha, intensity, batch_norm=True):
        super(GraphNet, self).__init__()

        # Nodes
        self.num_nodes = num_nodes  # Number of nodes in graph
        self.input_node_size = input_node_size  # Dimension of input node features
        self.hidden_node_size = output_node_size  # Dimension of hidden node features
        self.num_hidden_node_layers = num_hidden_node_layers  # Node layers in node networks

        # Edges
        self.hidden_edge_size = hidden_edge_size  # Hidden size in edge networks
        self.output_edge_size = output_edge_size  # Output size in edge networks
        self.input_edge_size = 2 * self.hidden_node_size

        self.intensity = intensity  # Whether the lasts element along the last axis are intensities (bool)

        if self.intensity:
            self.input_edge_size += 2  # Extra edge for relative positions and intensitities
            # assert self.hidden_node_size > 1, f"hidden_node_size is set to {self.hidden_node_size} <= 1: A feature cannot be both position and intensity!"

        self.num_mp = num_mp  # Number of message passing
        self.batch_norm = batch_norm  # Use batch normalization (bool)

        self.alpha = alpha  # For leaky relu layer for edge features
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for edge features

        # Edge networks
        self.edge_net_1 = nn.ModuleList()
        self.edge_net_2 = nn.ModuleList()

        # Nodes networks
        self.node_net_1 = nn.ModuleList()
        self.node_net_2 = nn.ModuleList()

        # Batch normalization layers
        self.bn_edge_1 = nn.ModuleList()
        self.bn_edge_2 = nn.ModuleList()
        self.bn_node = nn.ModuleList()

        for i in range(self.num_mp):
            # Edge feature layers
            self.edge_net_1.append(nn.Linear(self.input_edge_size, self.hidden_edge_size))
            self.edge_net_2.append(nn.Linear(self.hidden_edge_size, self.output_edge_size))

            if batch_norm:
                self.bn_edge_1.append(nn.BatchNorm1d(self.hidden_edge_size))
                self.bn_edge_2.append(nn.BatchNorm1d(self.output_edge_size))

            # Node feature layers
            node_layers = nn.ModuleList()
            node_layers.append(nn.Linear(self.output_edge_size + self.hidden_node_size, self.hidden_node_size))

            for j in range(self.num_hidden_node_layers - 1):
                node_layers.append(nn.Linear(self.hidden_node_size, self.hidden_node_size))

            self.node_net_1.append(node_layers)  # Layer for message Aggregation of hidden layer
            self.node_net_2.append(nn.Linear(self.hidden_node_size, self.hidden_node_size))  # Layer for message Aggregation

            if batch_norm:
                bn_node_i = nn.ModuleList()
                for i in range(num_hidden_node_layers):
                    bn_node_i.append(nn.BatchNorm1d(self.hidden_node_size))
                self.bn_node.append(bn_node_i)

    """
    Parameter
    ----------
    x: torch.Tensor
        The input node features.

    Return
    ------
    x: torch.Tensor
        Node features after message passing.
    """
    def forward(self, x):
        self.x = x
        batch_size = x.shape[0]

        x = F.pad(x, (0, self.hidden_node_size - self.input_node_size, 0, 0, 0, 0))

        for i in range(self.num_mp):
            # Edge features
            A = self.getA(x, batch_size)

            # Edge layer 1
            A = F.leaky_relu(self.edge_net_1[i](A), negative_slope=self.alpha)
            if self.batch_norm:
                A = self.bn_edge_1[i](A)
            A = self.dropout(A)

            # Edge layer 2
            A = F.leaky_relu(self.edge_net_2[i](A), negative_slope=self.alpha)
            if self.batch_norm:
                A = self.bn_edge_2[i](A)
            A = self.dropout(A)

            # Concatenation
            A = A.view(batch_size, self.num_nodes, self.num_nodes, self.output_edge_size)
            A = torch.sum(A, 2)
            x = torch.cat((A, x), 2)
            x = x.view(batch_size * self.num_nodes, self.output_edge_size + self.hidden_node_size)

            # Aggregation
            for j in range(self.num_hidden_node_layers):
                x = F.leaky_relu(self.node_net_1[i][j](x), negative_slope=self.alpha)
                if self.batch_norm:
                    x = self.bn_node[i][j](x)
                x = self.dropout(x)

            x = self.dropout(torch.tanh(self.node_net_2[i](x)))
            x = x.view(batch_size, self.num_nodes, self.hidden_node_size)

        return x

    """
    Parameters
    ----------
    x: torch.Tensor
        The node features.
    batch_size: int
        The batch size.

    Return
    ------
    A: torch.Tensor with shape (batch_size * self.num_nodes * self.num_nodes, self.input_edge_size)
        The adjacency matrix.
        As an example, for a specific batch,
        x[batch_idx] = [[x1, y1, I1],
                        [x2, y2, I2],
                        ...
                        [xn, yn, In]]
        If self.intensity is True, the adjacency for a specific batch has a format
        A[batch_idx] = [[x1, y1, I1, x1, y1, I1, d11, I11],
                        [x1, y1, I1, x2, y2, I2, d12, I12],
                        ...
                        [x1, y1, In, xn, yn, In, d1n, I1n],
                        [x1, y1, I1, x1, y1, I1, d11, I11],
                        [x1, y1, I1, x2, y2, I2, d12, I12],
                        ...
                        [x2, y2, I1, x1, y1, I1, d21, I21],
                        ...
                        [xn, yn, In, xn, yn, In, dnn, Imn]],
        where the relative intensity is defined by I_{ij} = 1 - (I_j - I_i).
        If self.intensity is False, the adjacency for a specific batch has a format
                        [x1, y1, I1, x2, y2, I2],
                        ...
                        [x1, y1, In, xn, yn, In],
                        [x1, y1, I1, x1, y1, I1],
                        [x1, y1, I1, x2, y2, I2],
                        ...
                        [x2, y2, I1, x1, y1, I1],
                        ...
                        [xn, yn, In, xn, yn, In]].
    """
    def getA(self, x, batch_size):
        x1 = x.repeat(1, 1, self.num_nodes).view(batch_size, self.num_nodes * self.num_nodes, self.hidden_node_size)
        x2 = x.repeat(1, self.num_nodes, 1) # 1*(self.num_nodes)*1 tensor with repeated x along axis=1

        if self.intensity:
            dists = torch.norm(x2[:, :, :-1] - x1[:, :, :-1] + 1e-12, dim=2).unsqueeze(2)
            int_diffs = 1 - ((x2[:, :, -1] - x1[:, :, -1])).unsqueeze(2)  # Iij = 1 - (Ij - Ii)
            A = (torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size * self.num_nodes * self.num_nodes, self.input_edge_size)
        else:
            A = torch.cat((x1, x2), 2).view(batch_size * self.num_nodes * self.num_nodes, self.input_edge_size)

        return A
