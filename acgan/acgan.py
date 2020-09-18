from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_size, num_classes):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        
        in_channels = self.in_size[0]

        first_modules = [
            # 3x32x32
            nn.Conv2d(in_channels, 64, kernel_size=3 , stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x16x16
            nn.Conv2d(64, 128, kernel_size=3 , stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x16x16
            nn.Conv2d(128, 256, kernel_size=3 , stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x8x8
            nn.Conv2d(256, 512, kernel_size=3 , stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512x8x8
            nn.Conv2d(512, 512, kernel_size=3 , stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512x4x4
        ]
        
        self.feature_extractor = nn.Sequential(*first_modules)
        
        # h, w = self.in_size[1:]
        # h_gag, w_gag = h // (2 ** 4), w // (2 ** 4)
        num_units = self.feature_extractor(torch.randn([1]+list(in_size))).flatten().size(0) 
        
        # second_modules = [nn.Linear(h_gag * w_gag * 256, 1)]
        features_to_score_modules = [nn.Linear(num_units, 1)]
        features_to_class_modules = [nn.Linear(num_units, self.num_classes)]
        
        self.features_to_score = nn.Sequential(*features_to_score_modules)
        self.features_to_class = nn.Sequential(*features_to_class_modules)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
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
        y_class = self.features_to_class(features.view(b_size, -1))
        
        # ========================
        return self.sigmoid(y), self.softmax(y_class)


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim # = 1024

        #  You can assume a fixed image size.
        fc_module = [nn.Linear(z_dim, 8*8*256)]
        self.fc = nn.Sequential(*fc_module)
        
        all_modules = [
            # 256x8x8
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False),
            # 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(256, 512, 4 ,stride=2, padding=1, bias=False),
            # 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 512, 3 ,stride=1, padding=1, bias=False),
            # 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, 4 ,stride=2, padding=1, bias=False),
            # 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(256, 128, 3 ,stride=1, padding=1, bias=False),
            # 128x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(128, out_channels, 3 ,stride=1, padding=1, bias=False),
            # 3x32x32
            nn.Tanh()
        ]

        self.gen = nn.Sequential(*all_modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        #  Generate n latent space samples and return their reconstructions.
        z = torch.randn(n, self.z_dim, device=device)
        cats = torch.randint(0, 10, (n,), device=device)
        onehot = torch.zeros((n, 10), device=device)
        onehot[np.arange(n), cats] = 1
        z[np.arange(n), :10] = onehot

        if with_grad:
            samples = self.forward(z)
        else:
            with torch.no_grad():
                samples = self.forward(z)
        # ========================
        return samples, cats
    
    def sample_cats(self, n, categories):
        """
        Samples from the Generator.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        #  Generate n latent space samples and return their reconstructions.
        z = torch.randn(n, self.z_dim, device=device)
        onehot = torch.zeros((n, 10), device=device)
        onehot[np.arange(n), categories] = 1
        z[np.arange(n), :10] = onehot

        with torch.no_grad():
            samples = self.forward(z)
        # ========================
        return samples, categories

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        b_size = z.size(0)
        output = self.fc(z)
        x = self.gen(output.view(b_size, 256, 8, 8))
        return x


def discriminator_loss_fn(y_data, y_data_class_soft, real_class_labels, 
                               y_generated, y_generated_class_soft, fake_class_labels,
                                data_label=0, label_noise=0.2):
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

    nll_fn = nn.NLLLoss()
    loss_class_real = nll_fn(torch.log(y_data_class_soft), real_class_labels)
    loss_class_fake = nll_fn(torch.log(y_generated_class_soft), fake_class_labels)

    # ========================
    return loss_data + loss_class_real + loss_class_fake


def generator_loss_fn(y_generated,y_generated_class_soft, fake_class_labels, data_label=0):
    """
    Computes the loss of the generator given generated data.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    loss = torch.mean((y_generated - data_label)**2)

    nll_fn = nn.NLLLoss()
    nll_loss = nll_fn(torch.log(y_generated_class_soft), fake_class_labels)
    return loss + nll_loss

def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader, y_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
   
    fake, gen_labels = gen_model.sample(x_data.shape[0], with_grad=True)

    dsc_optimizer.zero_grad()
    # Descriminator on real data
    real_output, real_classes = dsc_model(x_data)#.view(-1)    
    # Descriminator on fake data
    fake_output, fake_classes = dsc_model(fake.detach())

    dsc_loss = dsc_loss_fn(real_output, real_classes, y_data,
                                fake_output, fake_classes, gen_labels)

    dsc_loss.backward()
    dsc_optimizer.step()
    # =======================

    # Train Generator
    gen_optimizer.zero_grad()
    fake_output, fake_classes = dsc_model(fake)
    gen_loss = gen_loss_fn(fake_output, fake_classes, gen_labels)
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
