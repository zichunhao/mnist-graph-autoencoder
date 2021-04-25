import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import pickle

from torch.utils.data import DataLoader
from MNISTGraphDataset import MNISTGraphDataset

'''
Convert an array of coordinates [[xi, yi, Ii]] to a 2-D image array.
'''
def generate_img_arr(coords, img_dim=28):
    coords = coords.detach().numpy()
    coords = np.array(coords, copy=True)
    # Denormalization
    coords[:, :2]  = (coords[:, :2] + 1) / 2 * img_dim - 1e-5  # x,y
    coords[:, -1] = (coords[:, -1] + 1) * 127.5  # Intesity

    img_arr = np.zeros((img_dim, img_dim))
    for pts in coords:
        x,y,I = pts
        x,y = int(x),int(y)
        img_arr[y,x] = I  # row = y, col = x
    return img_arr

'''
Save generated image from an array in terms of coordinates.
'''
def save_img(img_arr, label, epoch, outpath):
    make_dir(outpath)

    plt.imshow(img_arr, cmap='gray')
    plt.savefig(f"{outpath}/epoch_{epoch}_num_{label}_{generate_id()}.png", dpi=600)
    plt.close()

'''
Make new directory if it does not exist.
'''
def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

'''
Generate a random hexadecimal ID for saving images.
'''
def generate_id():
    return uuid.uuid4().hex

'''
Save generated images.
'''
def save_gen_imgs(gen_imgs, labels, epoch, is_train, outpath):
    make_dir(f"{outpath}/generated_images/train")
    make_dir(f"{outpath}/generated_images/valid")
    for i in range(len(gen_imgs)):
        img_arr = generate_img_arr(gen_imgs[i])
        img_label = labels[i].argmax(dim=-1).item()
        if is_train:
            save_img(img_arr, label=img_label, epoch=epoch, outpath=f"{outpath}/generated_images/train")
        else:
            save_img(img_arr, label=img_label, epoch=epoch, outpath=f"{outpath}/generated_images/valid")

'''
Save data like losses and dts.
'''
def save_data(data, data_name, epoch, is_train, outpath, global_data=False):
    make_dir(f"{outpath}/model_evaluations/pkl_files")
    if not global_data:
        if is_train:
            with open(f'{outpath}/model_evaluations/pkl_files/train_{data_name}_epoch_{epoch}.pkl', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(f'{outpath}/model_evaluations/pkl_files/valid_{data_name}_epoch_{epoch}.pkl', 'wb') as f:
                pickle.dump(data, f)
    else:
        if is_train:
            with open(f'{outpath}/model_evaluations/pkl_files/train_{data_name}.pkl', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(f'{outpath}/model_evaluations/pkl_files/valid_{data_name}.pkl', 'wb') as f:
                pickle.dump(data, f)
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
Plot evaluation results
'''
def plot_eval_results(args, data, data_name, outpath):
    make_dir(f"{outpath}/model_evaluations")
    if args.load_to_train:
        start = args.load_epoch + 1
        end = start + args.num_epochs
    else:
        start = 1
        end = args.num_epochs

    x = [i for i in range(start, end+1)]

    if type(data) in [tuple, list] and len(data) == 2:
        train, valid = data
        plt.plot(x, train, label='Train')
        plt.plot(x, valid, label='Valid')
    else:
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
    return f"MnistAutoencoder_lr_{args.lr}_numEpochs_{args.num_epochs}_batchSize_{args.batch_size}_latentNodeSize_{args.latent_node_size}"
