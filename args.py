import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='MNIST Graph autoencoder options')

    # Data loading options
    parser.add_argument('--file-path', type=str, default='./mnist_data/', metavar='N',
                        help='The path of the data.')
    parser.add_argument('--fig-size', type=int, default=28, metavar='N',
                        help='The size of the figure. Default: 28 (for MNIST figures).')

    # Model options
    parser.add_argument('--num-nodes', type=int, default=100, metavar='N',
                        help='Number of nodes in both encoder and decoder. This is also the number of brightest points to keep.')
    parser.add_argument('--input-node-size', type=int, default=3, metavar='N',
                        help='Dimension of node features. Default: 3, for (x,y,I).')
    parser.add_argument('--latent-node-size', type=int, default=32, metavar='N',
                        help='Dimension of latent node features.')
    parser.add_argument('--num-hidden-node-layers', type=int, default=2, metavar='N',
                        help='The number of hidden node layers.')
    parser.add_argument('--hidden-edge-size', type=int, default=32, metavar='N',
                        help='Dimension of hidden edge features.')
    parser.add_argument('--output-edge-size', type=int, default=32, metavar='N',
                        help='Dimension of output edge features.')
    parser.add_argument('--num-mps', type=int, default=3, metavar='N',
                        help='Number of message passing.')
    parser.add_argument('--dropout', type=int, default=0.3, metavar='N',
                        help='Dropout value for edge features.')
    parser.add_argument('--alpha', type=int, default=0.3, metavar='N',
                        help='Alpha value for the leaky relu layer in edge network.')
    parser.add_argument('--intensity', type=bool, default=True, metavar='N',
                        help='Whether the last elements in output vectors are intensities. Default: True.')
    parser.add_argument('--batch-norm', type=bool, default=True, metavar='N',
                        help='Whether to include batch normalizations in the graph. Default: True.')

    # Training options
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=32, metavar='N',
                        help='Number of epochs for training.')

    args = parser.parse_args()

    return args
