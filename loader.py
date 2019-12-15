import cv2
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from config import *


def load_imgs(show_imgs=True):
    """
    Reads the images from the data dir specified in config.py
    :param show_imgs: Flag to display a sample of images
    :return: Dataloader containing the images
    """
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    if show_imgs:
        real_batch = next(iter(dataloader))

        sample_images = np.transpose(
            vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu().numpy(),
            (1, 2, 0))

        cv2.imshow('Data', cv2.cvtColor(sample_images, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dataloader


load_imgs()
