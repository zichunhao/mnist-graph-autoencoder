import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

from torch.utils.data import DataLoader
from MNISTGraphDataset import MNISTGraphDataset

'''
Convert an array of coordinates [[xi, yi, Ii]] to a 2-D image array.
'''
def generate_img_arr(coords, img_dim=28):
    coords = coords.detach().numpy()
    coords = np.array(coords, copy=True)
    # Denormalization
    coords[:, :2]  = (coords[:, :2] + 1) / 2 * img_dim - 1e-5
    coords[:, -1] = coords[:, -1] * 255 + 127.5

    img_arr = np.zeros((img_dim, img_dim))
    for pts in coords:
        x,y,I = pts
        x,y = int(x),int(y)
        img_arr[y,x] = I  # row = y, col = x
    return img_arr

'''
Save generated image from an array in terms of coordinates.
'''
def save_img(img_arr, label, epoch, save_dir):
    make_dir(save_dir)

    plt.imshow(img_arr, cmap='gray')
    plt.savefig(f"{save_dir}/{label}_{generate_id()}.png", dpi=600)
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
def save_gen_imgs(args, gen_imgs, labels, epoch, is_train):
    for i in range(len(gen_imgs)):
        img_arr = generate_img_arr(gen_imgs[i])
        img_label = labels[i].argmax(dim=-1).item()
        if is_train:
            save_img(img_arr, label=img_label, epoch=epoch, save_dir=f"{args.save_dir}/generated_images/train_epoch_{epoch}")
        else:
            save_img(img_arr, label=img_label, epoch=epoch, save_dir=f"{args.save_dir}/generated_images/valid_epoch_{epoch}")

'''
Save data like losses and dts.
'''
def save_data(args, data, data_name, epoch, is_train, global_data=False):
    if not global_data:
        if is_train:
            with open(f'{args.outpath}/model_evaluations/train_{data_name}_epoch_{epoch}.pkl', 'wb'):
                pickle.dump(data, f)
        else:
            with open(f'{args.outpath}/model_evaluations/valid_{data_name}_epoch_{epoch}.pkl', 'wb'):
                pickle.dump(data, f)
    else:
        if is_train:
            with open(f'{args.outpath}/model_evaluations/train_{data_name}.pkl', 'wb'):
                pickle.dump(data, f)
        else:
            with open(f'{args.outpath}/model_evaluations/valid_{data_name}.pkl', 'wb'):
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
