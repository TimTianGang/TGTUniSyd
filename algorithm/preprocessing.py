import numpy as np
import random
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import umap
import pickle
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, TensorDataset



def load_data(data_name, path='./data/'):  
    dataset = np.load(f'{path}{data_name}')
    X_train, Y_train = dataset['Xtr'], dataset['Str']
    X_test, Y_test = dataset['Xts'], dataset['Yts']
    return X_train, Y_train, X_test, Y_test


########## Preprocessing ##########

def transfer_gray_to_color(images):
    return np.stack((images,)*3, axis=1) 

def resize(images, data_name="mnist05"):

    if data_name.lower() in ("mnist05", "mnist06"):
        return images.reshape(images.shape[0], 1, 28, 28)
    
    elif data_name.lower() == "cifar":
        return images.reshape(images.shape[0], 3, 32, 32)
    
    else:
        print("Wrong dataset name!")
        return 

def transfer_unit_float(images):
    
    return images.astype(np.float32) #np.float默认64 而tensor.float默认64,先转为np.float32再传入model更安全


def normalize_channel(X): # also can implement by batch_norm of torch.nn
    """
    Normalize a dataset per channel (e.g RBG channels)
    input dim: (N, 1, W, H) for mnist and  (N, 3, W, H) for cifar
    """
    means, stds = [], []
    for channel in range(X.shape[1]):
        mean, std = X[:, channel, :, :].mean(), X[:, channel, :, :].std()
        X[:, channel, :, :] = (X[:, channel, :, :] - mean) / std
        means.append(mean)
        stds.append(std)
    #return X, {"mean": means, "std": stds}
    return X   

def unnormalize(X, mean, std):
    
    X = X * np.array(std).reshape(X.shape[0], 1, 1) + np.array(mean).reshape(X.shape[0], 1, 1)
    return X

def get_training_validation_samplers(trainset, validation_proportion=0.1):
    indices = np.arange(len(trainset))
    np.random.shuffle(indices)
    split = int((1 - validation_proportion) * len(trainset))

    train_sampler, validation_sampler = SubsetRandomSampler(indices[:split]), SubsetRandomSampler(indices[split:])

    return train_sampler, validation_sampler


def get_priors(Y):
    '''
    Estimation of priors for a set of labels
    '''
    priors = []
    classes = set(Y)
    masked = [i for i in Y]
    for c in classes:
        class_mask = (Y == c)
        priors.append(len(Y[class_mask]) / len(Y))
    return priors

def get_transitions(data_name="mnist05"):
    if data_name.lower() == "mnist05":
        return np.array([[0.5, 0.2, 0.3],
                        [0.3, 0.5, 0.2],
                        [0.2, 0.3, 0.5]])
    elif data_name.lower() == "mnist06":
        return np.array([[0.4, 0.3, 0.3],
                        [0.3, 0.4, 0.3],
                        [0.3, 0.3, 0.4]])
    else:
        print("Wrong name of dataset.")

########## Visualization ##########

def use_svg_display():
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_single_image(images, labels):
    #use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        #f.imshow(img.view((28, 28)).numpy())
        f.imshow(img)
        #f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def show_data(images, labels):
    for i in range(6):
        X, y = [], []
        for j in range(6):
            #print(i, j)
            X.append(images[i * 6 + j])
            y.append(labels[i * 6 + j])
        show_single_image(X, get_labels(y))
        #plt.savefig(data_name + ".jpg")

def show_grid_images(images, title=''):
    """
    From a list of images,
    Plots all images along a kx5 grid.
    """
    nbr_images = images.shape[0]
    figsize_ref = 1
    rows, columns = int(nbr_images / 5), 5
    f = plt.figure(figsize=(figsize_ref*columns,figsize_ref*rows)) # 2 Columns, . Lines
    plt.axis('off')
    plt.title(title,  y=1.08) # Trick to offset the title.
    plt.subplots_adjust(wspace=-0.35, hspace=0)
    for index in range(nbr_images):
        image = images[index]
        ax = f.add_subplot(rows, columns, index + 1)
        ax.axis('off')
        ax.imshow(image, cmap=plt.cm.gray)
    
    plt.show()
    

def serialize_object(obj, file_name):
    with open(file_name, 'wb') as file_handler:
        pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickled(file_name):
    pickled_data = {}
    if os.path.getsize(file_name) > 0:      
        with open(file_name, "rb") as f:
            unpickler = pickle.Unpickler(f)
            pickled_data = unpickler.load()
    return pickled_data