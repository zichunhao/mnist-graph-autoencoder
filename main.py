import json
import logging

import torch
from args import setup_argparse

from models.Encoder import Encoder
from models.Decoder import Decoder
from utils.utils import initialize_data, make_dir, gen_fname, plot_eval_results
from train import train_loop

if __name__ == "__main__":
    args = setup_argparse()

    '''Loggings'''
    if args.print_logging:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Num of GPUs: {torch.cuda.device_count()}")

    if device.type == 'cuda':
        print(f"GPU tagger: {torch.cuda.current_device()}")
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    print(f'Working on {str(device).upper()}')

    '''Initializations'''
    # Initialize and load dataset
    train_loader, valid_loader = initialize_data(args)

    # Initialize models
    encoder = Encoder(num_nodes=args.num_nodes, node_size=args.inputNodeSize,
                      latent_node_size=args.latentNodeSize, num_hidden_node_layers=args.num_hiddenNodeLayers,
                      hidden_edge_size=args.hiddenEdgeSize, output_edge_size=args.outputEdgeSize, num_mps=args.num_mps,
                      dropout=args.dropout, alpha=args.alpha, intensity=args.intensity, batch_norm=args.batch_norm, device=device).to(device)

    decoder = Decoder(num_nodes=args.num_nodes, node_size=args.inputNodeSize,
                      latent_node_size=args.latentNodeSize, num_hidden_node_layers=args.num_hiddenNodeLayers,
                      hidden_edge_size=args.hiddenEdgeSize, output_edge_size=args.outputEdgeSize, num_mps=args.num_mps,
                      dropout=args.dropout, alpha=args.alpha, intensity=args.intensity, batch_norm=args.batch_norm, device=device).to(device)

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        print('The models are initialized on GPU...')
    # One on cpu and the other on gpu
    elif (next(encoder.parameters()).is_cuda or next(encoder.parameters()).is_cuda):
        raise AssertionError("The encoder and decoder are not trained on the same device!")
    # Both on cpu
    else:
        print('The models are initialized on CPU...')

    print(f'Training over {args.num_epochs} epochs...')

    '''Training'''
    # Load existing model
    if args.load_toTrain:
        assert (args.load_epoch is not None), 'Which epoch weights to load is not specified!'
        outpath = args.load_modelPath
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load(f'{outpath}/weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'))
            decoder.load_state_dict(torch.load(f'{outpath}/weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth'))
        else:
            encoder.load_state_dict(torch.load(f'{outpath}/weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth', map_location=torch.device('cpu')))
            encoder.load_state_dict(torch.load(f'{outpath}/weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth', map_location=torch.device('cpu')))
    # Create new model
    else:
        outpath = f"{args.save_dir}/{gen_fname(args)}"

    make_dir(outpath)
    with open(f"{outpath}/args_cache.json", "w") as f:
        json.dump(vars(args), f)

    # Training
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)
    train_avg_losses, valid_avg_losses, train_dts, valid_dts = train_loop(args, encoder, decoder, train_loader, valid_loader,
                                                                          optimizer_encoder, optimizer_decoder, outpath, device=device)

    '''Plotting evaluation results'''
    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses), data_name="Losses", outpath=outpath)
    plot_eval_results(args, data=(train_dts, valid_dts), data_name="Time durations", outpath=outpath)
    plot_eval_results(args, data=[train_dts[i] + valid_dts[i] for i in range(len(train_dts))], data_name="Total time durations per epoch", outpath=outpath)
