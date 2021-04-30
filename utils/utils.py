import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import torch

from torch.utils.data import DataLoader
from utils.MNISTGraphDataset import MNISTGraphDataset

'''
Save generated image from an array in terms of coordinates.
'''
def save_img(img_arr, label, epoch, outpath, original=None):
    make_dir(outpath)
    rand_id = generate_id()

    # Generated image alone
    xGen = img_arr[:,0].reshape(-1)
    yGen = -1 * img_arr[:,1].reshape(-1)
    IGen = img_arr[:,2].reshape(-1)
    fig, ax = plt.subplots(1)
    ax.scatter(xGen, yGen, c=IGen, s=IGen*50)
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    ax.set_facecolor('black')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))
    plt.gray()
    plt.title(f"Number {label} at epoch {epoch + 1}")
    plt.savefig(f"{outpath}/epoch_{epoch+1}_num_{label}_{rand_id}.png", dpi=600)
    plt.close()

    # Generated image vs original image
    if original is not None:
        xOriginal = original[:,0].reshape(-1)
        yOriginal = -1 * original[:,1].reshape(-1)
        IOriginal = original[:,2].reshape(-1)

        make_dir(f"{outpath}/comparisons")
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].scatter(xOriginal, yOriginal, c=IOriginal, s=IOriginal*50)
        axes[0].set_xlim((-1,1))
        axes[0].set_ylim((-1,1))
        axes[0].set_facecolor('black')
        axes[0].set_xticks(np.arange(-1, 1.1, 0.5))
        axes[0].set_yticks(np.arange(-1, 1.1, 0.5))
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].set_title('original')
        axes[1].scatter(xGen, yGen, c=IGen, s=IGen*30)
        axes[1].set_xlim((-1,1))
        axes[1].set_ylim((-1,1))
        axes[1].set_facecolor('black')
        axes[1].set_xticks(np.arange(-1, 1.1, 0.5))
        axes[1].set_yticks(np.arange(-1, 1.1, 0.5))
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].set_title('generated')
        fig.suptitle(f"Number {label} at epoch {epoch+1}")
        plt.savefig(f"{outpath}/comparisons/epoch_{epoch+1}_num_{label}_{rand_id}.png", dpi=600, transparent=True)
        plt.close()

'''
Save generated images.
'''
def save_gen_imgs(gen_imgs, labels, epoch, is_train, outpath, originals=None):
    make_dir(f"{outpath}/generated_images/train")
    make_dir(f"{outpath}/generated_images/valid")
    for i in range(len(gen_imgs)):
        # Generated images
        if isinstance(gen_imgs[0], torch.Tensor):
            img_arr = gen_imgs[i].detach().cpu().numpy()
        else:
            img_arr = gen_imgs[i]

        # Labels (0-9)
        img_label = labels[i].argmax(dim=-1).item()

        # Original images
        if originals is not None:
            if isinstance(originals[0], torch.Tensor):
                original = originals[i].detach().cpu().numpy()
            else:
                original = originals[i]
        else:
            original = None
        if is_train:
            save_img(img_arr, label=img_label, epoch=epoch, outpath=f"{outpath}/generated_images/train", original=original)
        else:
            save_img(img_arr, label=img_label, epoch=epoch, outpath=f"{outpath}/generated_images/valid", original=original)


'''
Plot evaluation results
'''
def plot_eval_results(args, data, data_name, outpath, global_data=True):
    make_dir(f"{outpath}/model_evaluations")
    if args.load_toTrain:
        start = args.load_epoch + 1
        end = start + args.num_epochs
    else:
        start = 1
        end = args.num_epochs

    # (train, label)
    if type(data) in [tuple, list] and len(data) == 2:
        train, valid = data
        if global_data:
            x = [i for i in range(start, end+1)]
        else:
            x = [start + i for i in range(len(train))]

        if isinstance(train, torch.Tensor):
            train = train.detach().cpu().numpy()
        if isinstance(valid, torch.Tensor):
            valid = valid.detach().cpu().numpy()
        plt.plot(x, train, label='Train')
        plt.plot(x, valid, label='Valid')
    # only one type of data (e.g. dt)
    else:
        if global_data:
            x = [i for i in range(start, end+1)]
        else:
            x = [start + i for i in range(len(train))]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        plt.plot(x, data)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(data_name)
    plt.title(data_name)
    save_name = "_".join(data_name.lower().split(" "))
    plt.savefig(f"{outpath}/model_evaluations/{save_name}.pdf")
    plt.savefig(f"{outpath}/model_evaluations/{save_name}.png", dpi=600)
    plt.close()


'''
Generate folder name
'''
def gen_fname(args):
    return f"MnistAutoencoder_lr_{args.lr}_numEpochs_{args.num_epochs}_batchSize_{args.batch_size}_latentNodeSize_{args.latentNodeSize}"

'''
Generate a random hexadecimal ID for saving images.
'''
def generate_id():
    return uuid.uuid4().hex

'''
Make new directory if it does not exist.
'''
def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

'''
Save data like losses and dts.
'''
def save_data(data, data_name, epoch, is_train, outpath, global_data=False, cpu=True):
    make_dir(f"{outpath}/model_evaluations/pkl_files")
    if cpu and isinstance(data, torch.Tensor):
        data = data.cpu()
    if not global_data:
        if is_train:
            torch.save(data, f'{outpath}/model_evaluations/pkl_files/train_{data_name}_epoch_{epoch+1}.pkl')
        else:
            torch.save(data, f'{outpath}/model_evaluations/pkl_files/valid_{data_name}_epoch_{epoch+1}.pkl')
    else:
        if is_train:
            torch.save(data, f'{outpath}/model_evaluations/pkl_files/train_{data_name}.pkl')
        else:
            torch.save(data, f'{outpath}/model_evaluations/pkl_files/valid_{data_name}.pkl')


'''
Data initialization.
'''
def initialize_data(args):
    data_train = MNISTGraphDataset(dataset_path=args.file_path, num_pts=args.num_nodes, train=True)
    data_test = MNISTGraphDataset(dataset_path=args.file_path, num_pts=args.num_nodes, train=False)

    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

'''
DEPRECATED: Convert a torch tensor of coordinates [[xi, yi, Ii]] to a 2-D image numpy array.
'''
def generate_img_arr(coords, img_dim=28):
    coords = coords.detach().cpu().numpy()
    coords = np.array(coords, copy=True)
    # Denormalization
    coords[:, :2]  = (coords[:, :2] + 1) / 2 * img_dim - 1e-5  # x,y
    coords[:, -1] = (coords[:, -1] + 1) * 127.5  # Intesity

    img_arr = np.zeros((img_dim, img_dim))
    for pts in coords:
        x,y,I = pts
        x,y = int(x), int(y)
        img_arr[y,x] = max(img_arr[y,x] + I, 255)  # row = y, col = x
    return img_arr
