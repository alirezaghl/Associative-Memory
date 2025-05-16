import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def load_mnist(batch_size, norm_factor=1):
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(
        root='./mnist_data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    trainset = list(iter(trainloader))

    testset = torchvision.datasets.MNIST(
        root='./mnist_data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    testset = list(iter(testloader))
    
    for i, (img, label) in enumerate(trainset):
        trainset[i] = (img.reshape(len(img), 784) / norm_factor, label)
    
    for i, (img, label) in enumerate(testset):
        testset[i] = (img.reshape(len(img), 784) / norm_factor, label)
    
    return trainset, testset


class BlurredMNIST:
    def __init__(self, sigma):
        
        self.sigma = sigma
    
    def blur_batch(self, batch_data):
       
        images, labels = batch_data
        batch_size = images.shape[0]
        
        images_reshaped = images.reshape(batch_size, 28, 28)
        
        blurred_images = torch.zeros_like(images)
        
        for i in range(batch_size):
            img_np = images_reshaped[i].numpy()
            
            blurred_img_np = gaussian_filter(img_np, sigma=self.sigma)
            
            blurred_images[i] = torch.tensor(blurred_img_np).reshape(784)
        
        return blurred_images, labels
    
    def blur_dataset(self, dataset):
        blurred_dataset = []
        for batch_data in dataset:
            blurred_batch = self.blur_batch(batch_data)
            blurred_dataset.append(blurred_batch)
        
        return blurred_dataset


def Attention(x, z, beta):
    return x.T @ F.softmax(beta * x @ z, dim=0)

def retrieve_store_continuous(imgs, Z, N, beta, num_plot):
  X = imgs[0:N,:]
  Z = Z[0:N, :]

  for j in range(num_plot):
    out = Attention(X,Z[j, :],beta)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    imgs = [X[j,:], Z[j, :], out]
    titles = ["Original","Masked","Reconstruction"]
    fig.suptitle(f"β = {beta}", fontsize=16)
    for i, ax in enumerate(axs.flatten()):
      plt.sca(ax)
      plt.imshow(imgs[i].reshape(28,28))
      plt.title(titles[i])
    plt.show()

trainset, testset = load_mnist(1000)
blurrer = BlurredMNIST(sigma=1)
blurred_trainset = blurrer.blur_dataset(trainset)
imgs, labels = trainset[0]
blurred_imgs, _ = blurred_trainset[0]
beta = [0.25, 0.5, 1, 4.0, 6.0]
N = 1000
NUM_PLOTS = 5

for b in beta:
    retrieve_store_continuous(imgs, blurred_imgs, N, b, NUM_PLOTS)