import torch

dataroot = "/d_drive/Datasets/celeba"
workers = 2
batch_size = 128
image_size = 64  # Spatial Size of images
nc = 3  # No of channels
nz = 100  # Size of z
ngf = 64  # Size of feature maps in G
ndf = 64  # Size of feature maps in D
num_epochs = 15
lr = 0.0002
beta1 = 0.5  # For Adam
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
