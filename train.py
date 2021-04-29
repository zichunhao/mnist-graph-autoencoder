import torch
import torch.nn as nn
import time

from utils.utils import make_dir, generate_img_arr, save_gen_imgs, save_data, plot_eval_results
# from utils.loss import chamfer_loss_batch
from utils.loss import ChamferLoss

def train(args, encoder, decoder, loader, epoch, optimizer_encoder, optimizer_decoder, outpath, is_train, device):
    epoch_total_loss = 0
    labels = []
    gen_imgs = []
    if args.compareFigs:
        original = []

    if is_train:
        encoder.train()
        decoder.train()

    for i, batch in enumerate(loader, 0):
        X, Y = batch[0].to(device), batch[1]
        batch_gen_imgs = decoder(encoder(X), args)

        loss = ChamferLoss(device)
        batch_loss = loss(batch_gen_imgs, X)
        epoch_total_loss += batch_loss

        # True if batch_loss has at least one NaN value
        if (batch_loss != batch_loss).any():
            raise RuntimeError('Batch loss is NaN!')

        # back prop
        if is_train:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            batch_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
        #     print(f"epoch {epoch+1}, batch {i+1}/{len(loader)}, train_loss={batch_loss.item()}", end='\r', flush=True)
        # else:
        #     print(f"epoch {epoch+1}, batch {i+1}/{len(loader)}, valid_loss={batch_loss.item()}", end='\r', flush=True)

        # Save all generated images
        if args.save_figs and args.save_allFigs:
            labels.append(Y.cpu())
            gen_imgs.append(torch.tanh(batch_gen_imgs).cpu())
            if args.compareFigs:
                original.append(X.cpu())

        # Save only the last batch
        elif args.save_figs:
            if (i == len(loader) - 1):
                labels.append(Y.cpu())
                gen_imgs.append(torch.tanh(batch_gen_imgs).cpu())
                if args.compareFigs:
                    original.append(X.cpu())

    # Save model
    if is_train:
        make_dir(f'{outpath}/weights_encoder')
        make_dir(f'{outpath}/weights_decoder')
        torch.save(encoder.state_dict(), f"{outpath}/weights_encoder/epoch_{epoch+1}_encoder_weights.pth")
        torch.save(decoder.state_dict(), f"{outpath}/weights_decoder/epoch_{epoch+1}_decoder_weights.pth")

    # Compute average loss
    epoch_avg_loss = epoch_total_loss / len(loader)
    save_data(epoch_avg_loss, "loss", epoch, is_train, outpath)

    for i in range(len(gen_imgs)):
        if args.compareFigs:
            save_gen_imgs(gen_imgs[i], labels[i], epoch, is_train, outpath, originals=original[i].cpu())
        else:
            save_gen_imgs(gen_imgs[i], labels[i], epoch, is_train, outpath)

    return epoch_avg_loss, gen_imgs

@torch.no_grad()
def test(args, encoder, decoder, loader, epoch, optimizer_encoder, optimizer_decoder, outpath, device):
    with torch.no_grad():
        epoch_avg_loss, gen_imgs = train(args, encoder, decoder, loader, epoch, optimizer_encoder, optimizer_decoder,
                                         outpath, is_train=False, device=device)
    return epoch_avg_loss, gen_imgs

def train_loop(args, encoder, decoder, train_loader, valid_loader, optimizer_encoder, optimizer_decoder, outpath, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (args.save_dir is not None), "Please specify directory of saving the models!"
    make_dir(args.save_dir)

    train_avg_losses = []
    train_dts = []
    valid_avg_losses = []
    valid_dts = []

    for ep in range(args.num_epochs):
        if args.load_toTrain:
            epoch = args.load_epoch + ep + 1
        else:
            epoch = ep

        # Training
        start = time.time()
        train_avg_loss, train_gen_imgs = train(args, encoder, decoder, train_loader, epoch,
                                               optimizer_encoder, optimizer_decoder, outpath, is_train=True, device=device)
        train_dt = time.time() - start

        train_avg_losses.append(train_avg_loss.cpu())
        train_dts.append(train_dt)

        save_data(data=train_avg_loss, data_name="loss", epoch=epoch, outpath=outpath, is_train=True)
        save_data(data=train_dt, data_name="dt", epoch=epoch, outpath=outpath, is_train=True)

        # Validation
        start = time.time()
        valid_avg_loss, valid_gen_imgs = test(args, encoder, decoder, valid_loader, epoch,
                                              optimizer_encoder, optimizer_decoder, outpath, device=device)
        valid_dt = time.time() - start

        valid_avg_losses.append(train_avg_loss.cpu())
        valid_dts.append(valid_dt)

        save_data(data=valid_avg_loss, data_name="loss", epoch=epoch, outpath=outpath, is_train=False)
        save_data(data=valid_dt, data_name="dt", epoch=epoch, outpath=outpath, is_train=False)

        print(f'epoch={epoch+1}/{args.num_epochs if not args.load_toTrain else args.num_epochs+args.load_epoch}, '
              + f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, dt={train_dt+valid_dt}')

        if (epoch > 0) and ((epoch + 1) % 10 == 0):
            plot_eval_results(args, (train_avg_losses, valid_avg_losses), f"losses_to_{epoch+1}", outpath)

    # Save global data
    save_data(data=train_avg_losses, data_name="losses", epoch="global", outpath=outpath, is_train=True, global_data=True)
    save_data(data=train_dts, data_name="dts", epoch="global", outpath=outpath, is_train=True, global_data=True)
    save_data(data=valid_avg_losses, data_name="losses", epoch="global", outpath=outpath, is_train=False, global_data=True)
    save_data(data=valid_dts, data_name="dts", epoch="global", outpath=outpath, is_train=False, global_data=True)

    return train_avg_losses, valid_avg_losses, train_dts, valid_dts
