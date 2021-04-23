import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

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
        img_arr[y,x] = I
    return img_arr

def save_img(img_arr, label, epoch, save_dir="./generated_imgs"):
    make_dir(save_dir)
    
    plt.imshow(img_arr, cmap='gray')
    plt.savefig(f"{save_dir}/{label}_{generate_rand()}.png", dpi=900)
    plt.close()

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def generate_id():
    return uuid.uuid4().hex
