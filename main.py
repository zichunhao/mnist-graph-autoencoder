import args
from args import setup_argparse
import json
import logging

import torch
from models.Autoencoder import Autoencoder
from utils import initialize_data, make_dir
from train import train_loop

if __name__ == "__main__":
    args = setup_argparse()

    '''Loggings'''
    if args.print_logging:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Num of GPUs: {torch.cuda.device_count()}")

    if device.type == 'cuda':
        print(f"GPU tagger: {torch.cuda.current_device()}")
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    print(f'Working on {str(device).upper()}')

    '''Initializations'''
    # Initialize and load dataset
    train_loader, valid_loader = initialize_data(args)

    # Initialize model
    model = Autoencoder(
        num_nodes=args.num_nodes, node_size=args.input_node_size,
        latent_node_size=args.latent_node_size, num_hidden_node_layers=args.num_hidden_node_layers,
        hidden_edge_size=args.hidden_edge_size, output_edge_size=args.output_edge_size, num_mps=args.num_mps,
        dropout=args.dropout, alpha=args.alpha, intensity=args.intensity, batch_norm=args.batch_norm
    )

    if (next(model.parameters()).is_cuda):
        print('The model is initialized on GPU...')
    else:
        print('The model is initialized on CPU...')

    '''Training'''
    # Toad existing model
    if args.load_to_train:
        outpath = args.load_model_path
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f'{outpath}/epoch_{args.load_epoch}_weights.pth'))
        else:
            model.load_state_dict(torch.load(f'{outpath}/epoch_{args.load_epoch}_weights.pth', map_location=torch.device('cpu')))
    # create new model
    else:
        outpath = args.save_dir

    make_dir(outpath)
    with open(f"{outpath}/args_cache.json", "w") as f:
        json.dump(vars(args), f)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    train_avg_losses, valid_avg_losses, train_dts, valid_dts = train_loop(args, model, train_loader, valid_loader, optimizer)
