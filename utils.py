import numpy as np
import matplotlib.pyplot as plt
import os

def generate_img_array(coords, img_dim=28):
    coords = coords.detach().numpy()
    coords = np.array(coords, copy=True)
    # Denormalization
    coords[:, :2]  = (coords[:, :2] + 1) / 2 * img_dim - 1e-5
    coords[:, -1] = coords[:, -1] * 255 + 127.5

    img_arr = np.zeros((img_dim, img_dim))
    for pts in coords:
        x,y,I = pts
        x,y = int(x),int(y)
        img_arr[y,x] = I
    return img_arr

def save_img(img_arr, label="unknown", epoch=None, save_dir="./generated_imgs"):
    fig, ax = plt.subplots()
    ax.imshow(img_arr, cmap='gray')
    if epoch is None:
        PATH = f"{save_dir}"
    else:
        PATH = f"{save_dir}/epoch_{epoch}"
    make_path(PATH)
    plt.savefig(f"{PATH}/{label}.png", dpi=900)
    plt.close()

def make_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
