import torch
import torch.nn as nn

from models.GraphNet import GraphNet

class Autoencoder(nn.Module):

    """
    The autoencoder for MNIST dataset. The input is in the format [[x1, y1, I1], ...],
    formatted by MNISTGraphDataset.py. There is one node in the latent space, resulted from
    summing over all node features across axis=-2.

    Parameters
    ----------
    num_nodes: int
        The number of nodes for the graph.
    node_size: int
        The dimension of input node feature vectors.
    latent_node_size: int
        The dimension of node feature in the latent space.
    num_hidden_node_layers: int
        The number of layers of hidden nodes.
    hidden_edge_size: int
        The dimension of hidden edges before message passing.
    output_edge_size: int
        The dimenson of output edges for message passing.
    num_mps: int
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
    def __init__(self, num_nodes, node_size, latent_node_size, num_hidden_node_layers, hidden_edge_size, output_edge_size,
                 num_mps, dropout, alpha, intensity, batch_norm=True, device=None):
        super(Autoencoder, self).__init__()

        self.num_nodes = num_nodes
        self.node_size = node_size
        self.num_latent_node = 1  # We are summing over all node features, resulting in one node feature
        self.latent_node_size = latent_node_size
        self.num_mps = num_mps
        self.intensity = intensity

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.encoder = GraphNet(num_nodes=self.num_nodes, input_node_size=node_size, output_node_size=self.latent_node_size,
                                num_hidden_node_layers=num_hidden_node_layers, hidden_edge_size=hidden_edge_size,
                                output_edge_size=output_edge_size, num_mps=num_mps, dropout=dropout, alpha=alpha,
                                intensity=self.intensity, batch_norm=batch_norm, device=self.device)

        self.linear = nn.Linear(self.latent_node_size, self.num_nodes*self.latent_node_siz, device=self.device)

        self.decoder = GraphNet(num_nodes=self.num_nodes, input_node_size=self.latent_node_size, output_node_size=self.node_size,
                                num_hidden_node_layers=num_hidden_node_layers, hidden_edge_size=hidden_edge_size,
                                output_edge_size=output_edge_size, num_mps=num_mps, dropout=dropout, alpha=alpha,
                                intensity=self.intensity, batch_norm=batch_norm, device=self.device)

    """
    The forward pass of the Autoencoder.

    Parameters
    ----------
    x: torch.Tensor
        The input feature vectors
        Shape: (batch_size, num_nodes, 3)

    Returns
    -------
    latent_vec: torch.Tensor
        The vector in latent space.
        Shape: (batch_size, 1, 3)
    x: torch.Tensor
        The feature vectors encoded and then decoded by the Autoencoder.
        Shape: (batch_size, num_nodes, 3)
    """
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        latent_vec = torch.sum(x, dim=-2).unsqueeze(dim=0)  # Latent vector
        x = self.linear(latent_vec).view(batch_size, self.num_nodes, self.latent_node_size)
        x = self.decoder(x)
        x = torch.tanh(x)
        return latent_vec, x
