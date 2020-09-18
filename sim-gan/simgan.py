from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        
        in_channels = self.in_size[0]

        first_modules = [
            # 3x36x60
            nn.Conv2d(in_channels, 64, kernel_size=1 , stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x36x60
            nn.Conv2d(64, 128, kernel_size=2 , stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 
            nn.Conv2d(128, 256, kernel_size=3 , stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 
            nn.Conv2d(256, 512, kernel_size=3 , stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 
            nn.Conv2d(512, 512, kernel_size=4 , stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512x3x6
        ]
        
        self.feature_extractor = nn.Sequential(*first_modules)
        
        num_units = self.feature_extractor(torch.randn([1]+list(in_size))).flatten().size(0) 
        features_to_score_modules = [nn.Linear(num_units, 1)]
        
        self.features_to_score = nn.Sequential(*features_to_score_modules)
        self.sigmoid = nn.Sigmoid()
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        b_size = x.shape[0]
        features = self.feature_extractor(x)
        y = self.features_to_score(features.view(b_size, -1))
        # ========================
        return y#self.sigmoid(y)


class Generator(nn.Module):
    def __init__(self, in_size, out_channels=1):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.in_size = in_size
        in_channels = self.in_size[0]
        
        all_modules = [
            # 
            nn.ConvTranspose2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            #
            nn.ConvTranspose2d(256, 512, 1 ,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            #
            nn.ConvTranspose2d(512, 256, 1 ,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            #
            nn.ConvTranspose2d(256, 128, 1 ,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #
            nn.ConvTranspose2d(128, 64, 1 ,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #
            nn.ConvTranspose2d(64, out_channels, 1 ,stride=1, padding=0, bias=False),
            nn.Tanh()
        ]

        self.gen = nn.Sequential(*all_modules)
        # ========================

    def forward(self, imgs):
        """
        :param imgs: A batch of syntethic iamges samples of shape (N, size).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        x = self.gen(imgs)
        
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.2):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a least squares.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    device = y_data.device
    b_size = y_data.size(0)

    loss_data = torch.mean(((y_data - data_label)**2) +  ((y_generated - (1 - data_label))**2))

    # ========================
    return loss_data


def generator_loss_fn(y_generated, x_generated, x_sim ,lam=0.2, data_label=0):
    """
    Computes the loss of the generator given generated data.
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    loss = torch.mean((y_generated - data_label)**2)

    loss_fn = nn.L1Loss()
    l1 = loss_fn(x_generated, x_sim)

    return loss + lam*l1

def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data_real: DataLoader, x_data_sim: DataLoader,):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
    gen_optimizer.zero_grad()
    dsc_optimizer.zero_grad()
    
    fake = gen_model(x_data_sim)
    
    # Descriminator on real data
    real_output = dsc_model(x_data_real)#.view(-1)    
    # Descriminator on fake data
    fake_output = dsc_model(fake.detach())

    dsc_loss = dsc_loss_fn(real_output, fake_output)

    dsc_loss.backward()
    dsc_optimizer.step()
    # =======================

    # Train Generator
    
    fake_output = dsc_model(fake)
    gen_loss = gen_loss_fn(fake_output, fake, x_data_sim)
    #
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()

def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    
    if len(dsc_losses) % 5 == 0:
        saved = True
        torch.save(gen_model, checkpoint_file)

    return saved