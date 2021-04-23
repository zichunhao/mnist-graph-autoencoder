import torch.nn as nn
import time

from utils import make_dir, generate_img_arr, save_img, save_gen_imgs, save_data, plot_eval_results

def train(args, model, loader, epoch, optimizer, is_train):
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

        # Save all generated images
        if args.save_figs and args.save_all_figs:
            labels.append(Y)
            gen_imgs.append(batch_gen_imgs)
        # Save only the last batch
        elif args.save_figs and not args.save_all_figs:
            if i == len(loader) - 1:
                labels.append(Y)
                gen_imgs.append(batch_gen_imgs)

    # Save generated images
    save_gen_imgs(args, gen_imgs, labels, epoch)

    # Compute average loss
    epoch_avg_loss = epoch_total_loss / len(loader)

    make_dir(path=f"{args.save_dir}/epoch_avg_loss")


    return epoch_avg_loss, gen_imgs

def train_loop(args, model, train_loader, valid_loader, optimizer):
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
        train_avg_loss, _ = train(args, model, train_loader, epoch, optimizer, is_train=True)
        train_dt = time.time() - start

        train_avg_losses.append(train_avg_loss)
        train_dts.append(train_dt)

        save_data(args, data=train_avg_loss, data_name="loss", epoch=epoch, is_train=True)
        save_data(args, data=train_dt, data_name="dt", epoch=epoch, is_train=True)

        # Validation
        start = time.time()
        valid_avg_loss, _ = train(args, model, valid_loader, epoch, optimizer, is_train=False)
        valid_dt = time.time() - start

        valid_avg_losses.append(train_avg_loss)
        valid_dts.append(valid_dt)

        save_data(args, data=valid_avg_loss, data_name="loss", epoch=epoch, is_train=False)
        save_data(args, data=valid_dt, data_name="dt", epoch=epoch, is_train=False)

        print(f'epoch={epoch+1}/{args.num_epochs if not args.load_to_train else args.num_epochs+args.load_epoch} '
            + f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, dt={train_dt+valid_dt}')

    # Save global data
    save_data(args, data=train_avg_losses, data_name="losses", epoch="global", is_train=True, global_data=True)
    save_data(args, data=train_dts, data_name="dts", epoch="global", is_train=True, global_data=True)
    save_data(args, data=valid_avg_losses, data_name="losses", epoch="global", is_train=False, global_data=True)
    save_data(args, data=valid_dts, data_name="dts", epoch="global", is_train=False, global_data=True)

    return train_avg_losses, valid_avg_losses, train_dts, valid_dts

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses), data_name="Losses")
    plot_eval_results(args, data=(train_dts, valid_dts), data_name="Time durations")
    plot_eval_results(args, data=[train_dts[i] + valid_dts[i]], data_name="Total time durations per epoch")
