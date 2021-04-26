import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='MNIST Graph autoencoder options')

    # Data loading options
    parser.add_argument('--file-path', type=str, default='./mnist_data/', metavar='',
                        help='The path of the data.')
    parser.add_argument('--fig-size', type=int, default=28, metavar='',
                        help='The size of the figure. Default: 28 (for MNIST figures).')

    # Model options
    parser.add_argument('--num-nodes', type=int, default=100, metavar='',
                        help='Number of nodes in both encoder and decoder. This is also the number of brightest points to keep.')
    parser.add_argument('--inputNodeSize', type=int, default=3, metavar='',
                        help='Dimension of node features. Default: 3, for (x,y,I).')
    parser.add_argument('--latentNodeSize', type=int, default=32, metavar='',
                        help='Dimension of latent node features.')
    parser.add_argument('--num-hiddenNodeLayers', type=int, default=2, metavar='',
                        help='The number of hidden node layers.')
    parser.add_argument('--hiddenEdgeSize', type=int, default=32, metavar='',
                        help='Dimension of hidden edge features.')
    parser.add_argument('--outputEdgeSize', type=int, default=32, metavar='',
                        help='Dimension of output edge features.')
    parser.add_argument('--num-mps', type=int, default=3, metavar='',
                        help='Number of message passing.')
    parser.add_argument('--dropout', type=int, default=0.3, metavar='',
                        help='Dropout value for edge features.')
    parser.add_argument('--alpha', type=int, default=0.3, metavar='',
                        help='Alpha value for the leaky relu layer in edge network.')
    parser.add_argument('--intensity', type=strToBool, default=True, metavar='',
                        help='Whether the last elements in output vectors are intensities. Default: True.')
    parser.add_argument('--batch-norm', type=strToBool, default=True, metavar='',
                        help='Whether to include batch normalizations in the graph. Default: True.')

    # Training options and hyperparameters
    parser.add_argument('--batch-size', type=int, default=8, metavar='',
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=64, metavar='',
                        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                        help='Learning rate.')
    parser.add_argument('--save-dir', type=str, default='mnist_trained_models', metavar='',
                        help='The directory to save trained models and figures.')
    parser.add_argument('--save-figs', type=strToBool, default=True, metavar='',
                        help='Whether to save generated figures.')
    parser.add_argument('--save-allFigs', type=strToBool, default=False, metavar='',
                        help='Whether to save figures generated in ALL epochs.')
    parser.add_argument('--load-toTrain', type=strToBool, default=False, metavar='',
                        help='Whether to load existing (trained) model for training.')
    parser.add_argument('--load-modelPath', type=str, default=None, metavar='',
                        help='Path of the trained model to load.')
    parser.add_argument('--load-epoch', type=int, default=None, metavar='',
                        help='Epoch number of the trained model to load.')

    parser.add_argument('--print-logging', type=strToBool, default=True, metavar='',
                        help='Whether to print logging infos.')

    args = parser.parse_args()

    return args

# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def strToBool(arg):
    if isinstance(arg, bool):
       return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
