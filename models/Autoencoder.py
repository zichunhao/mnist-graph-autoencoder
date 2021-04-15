import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GraphNet import GraphNet

class Autoencoder(nn.Module):

    def __init__(self, num_nodes, node_size, latent_dim, num_hidden_node_layers, hidden_edge_size, output_edge_size,
                 num_mp, dropout, alpha, intensity, batch_norm=True):
        super(Autoencoder, self).__init__()

        self.num_nodes = num_nodes
        self.node_size = node_size
        self.num_latent_node = 1  # We are summing over all node features
        self.latent_dim = latent_dim
        self.num_mp = num_mp
        self.intensity = intensity

        self.encoder = GraphNet(num_nodes=self.num_nodes, input_node_size=node_size, output_node_size=self.latent_dim,
                                num_hidden_node_layers=num_hidden_node_layers, hidden_edge_size=hidden_edge_size,
                                output_edge_size=output_edge_size, num_mp=num_mp, dropout=dropout, alpha=alpha,
                                intensity=self.intensity, batch_norm=batch_norm)

        self.linear = nn.Linear(self.latent_dim, self.num_nodes*self.latent_dim)

        self.decoder = GraphNet(num_nodes=self.num_nodes, input_node_size=self.latent_dim, output_node_size=self.node_size,
                                num_hidden_node_layers=num_hidden_node_layers, hidden_edge_size=hidden_edge_size,
                                output_edge_size=output_edge_size, num_mp=num_mp, dropout=dropout, alpha=alpha,
                                intensity=self.intensity, batch_norm=batch_norm)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        latent_vec = torch.sum(x, dim=-2).unsqueeze(dim=0)  # Latent vector
        x = self.linear(latent_vec).view(batch_size, self.num_nodes, self.latent_dim)
        x = self.decoder(x)
        return latent_vec, x
