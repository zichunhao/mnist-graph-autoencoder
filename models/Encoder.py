import torch
import torch.nn as nn

from models.GraphNet import GraphNet

class Encoder(nn.Module):
    def __init__(self, num_nodes, node_size, latent_node_size, num_hidden_node_layers, hidden_edge_size, output_edge_size,
                 num_mps, dropout, alpha, intensity, batch_norm=True, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.node_size = node_size
        self.num_latent_node = 1  # We are summing over all node features, resulting in one node feature
        self.latent_node_size = latent_node_size
        self.num_mps = num_mps
        self.intensity = intensity

        self.device = device

        # layers
        self.encoder = GraphNet(num_nodes=self.num_nodes, input_node_size=node_size, output_node_size=self.latent_node_size,
                                num_hidden_node_layers=num_hidden_node_layers, hidden_edge_size=hidden_edge_size,
                                output_edge_size=output_edge_size, num_mps=num_mps, dropout=dropout, alpha=alpha,
                                intensity=self.intensity, batch_norm=batch_norm, device=self.device).to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        latent_vec = torch.sum(x, dim=-2).unsqueeze(dim=0)  # Latent vector
        return latent_vec
