import torch
import torch.nn as nn
import time

from utils import make_dir, generate_img_arr, save_img, save_gen_imgs, save_data

def train(args, model, loader, epoch, optimizer, outpath, is_train):
    epoch_total_loss = 0
    labels = []
    gen_imgs = []

    for i, batch in enumerate(loader, 0):
        X, Y = batch
        _, batch_gen_imgs = model(X)  # batch_latent_vecs, batch_gen_imgs

        loss = nn.MSELoss()
        batch_loss = loss(batch_gen_imgs, X)
        epoch_total_loss += batch_loss
        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print(f"epoch {epoch}, batch {i+1}/{len(loader)}, train_loss={batch_loss.item()}", end='\r', flush=True)
        else:
            print(f"epoch {epoch}, batch {i+1}/{len(loader)}, valid_loss={batch_loss.item()}", end='\r', flush=True)

        labels.append(Y)
        gen_imgs.append(batch_gen_imgs)

    # Compute average loss
    epoch_avg_loss = epoch_total_loss / len(loader)

    make_dir(path=f"{args.save_dir}/epoch_avg_loss")

    return epoch_avg_loss, gen_imgs

def train_loop(args, model, train_loader, valid_loader, optimizer, outpath):
    assert (args.save_dir is not None), "Please specify save directory!"
    make_dir(args.save_dir)

    train_avg_losses = []
    train_dts = []
    valid_avg_losses = []
    valid_dts = []

    for ep in range(args.num_epochs):
        if args.load_to_train:
            epoch = args.load_epoch + ep + 1
        else:
            epoch = ep

        # Training
        start = time.time()
        train_avg_loss, train_gen_imgs = train(args, model, train_loader, epoch, optimizer, outpath, is_train=True)
        train_dt = time.time() - start

        # Save model
        make_dir(f'{outpath}/weights')
        torch.save(model.state_dict(), f"{outpath}/weights/epoch_{epoch+1}_weights.pth")

        train_avg_losses.append(train_avg_loss)
        train_dts.append(train_dt)

        save_data(data=train_avg_loss, data_name="loss", epoch=epoch, outpath=outpath, is_train=True)
        save_data(data=train_dt, data_name="dt", epoch=epoch, outpath=outpath, is_train=True)

        # Validation
        start = time.time()
        valid_avg_loss, valid_gen_imgs = train(args, model, valid_loader, epoch, optimizer, outpath, is_train=False)
        valid_dt = time.time() - start

        valid_avg_losses.append(train_avg_loss)
        valid_dts.append(valid_dt)

        save_data(data=valid_avg_loss, data_name="loss", epoch=epoch, outpath=outpath, is_train=False)
        save_data(data=valid_dt, data_name="dt", epoch=epoch, outpath=outpath, is_train=False)

        # Save all generated images
        if args.save_figs and args.save_all_figs:
            for i in range(len(train_gen_imgs)):
                save_gen_imgs(train_gen_imgs[i], labels, epoch, is_train=True, outpath)
                save_gen_imgs(valid_gen_imgs[i], labels, epoch, is_train=False, outpath)
        # Save only the last batch
        else:
            save_gen_imgs(train_gen_imgs[-1], labels, epoch, is_train=True, outpath)
            save_gen_imgs(valid_gen_imgs[-1], labels, epoch, is_train=False, outpath)

        print(f'epoch={epoch+1}/{args.num_epochs if not args.load_to_train else args.num_epochs+args.load_epoch} '
            + f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, dt={train_dt+valid_dt}')

    # Save global data
    save_data(data=train_avg_losses, data_name="losses", epoch="global", outpath=outpath, is_train=True, global_data=True)
    save_data(data=train_dts, data_name="dts", epoch="global", outpath=outpath, is_train=True, global_data=True)
    save_data(data=valid_avg_losses, data_name="losses", epoch="global", outpath=outpath, is_train=False, global_data=True)
    save_data(data=valid_dts, data_name="dts", epoch="global", outpath=outpath, is_train=False, global_data=True)

    return train_avg_losses, valid_avg_losses, train_dts, valid_dts
