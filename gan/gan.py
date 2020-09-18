from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


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
            nn.Conv2d(in_channels, 64, kernel_size=5 , stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5 , stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5 , stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        ]
        
        self.feature_extractor = nn.Sequential(*first_modules)
        
        h, w = self.in_size[1:]
        h_gag, w_gag = h // (2 ** 3), w // (2 ** 3)
        
        second_modules = [nn.Linear(h_gag * w_gag * 256, 1)]
        self.features_to_score = nn.Sequential(*second_modules)
        
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
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        #  You can assume a fixed image size.
        
        all_modules = [
            nn.ConvTranspose2d(self.z_dim, 1024, featuremap_size,stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, featuremap_size,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, featuremap_size,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, featuremap_size,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, out_channels, featuremap_size,stride=2, padding=1, bias=False),
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
        #  Don't use a loop.
        z = torch.randn(n, self.z_dim, device=device)
        if with_grad:
            samples = self.forward(z)
        else:
            with torch.no_grad():
                samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        x = self.gen(torch.unsqueeze(torch.unsqueeze(z, dim=2), dim=3))
        
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    #  Pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    device = y_data.device
    
    n_y_data =  torch.distributions.uniform.Uniform(-label_noise / 2, label_noise / 2).sample(y_data.size()).to(device) + data_label
    n_y_gen = torch.distributions.uniform.Uniform(-label_noise / 2, label_noise / 2).sample(y_data.size()).to(device) + (1 - data_label)
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_data = loss_fn(y_data, n_y_data)
    loss_generated = loss_fn(y_generated, n_y_gen)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    data_label_tensor = torch.ones(y_generated.shape, dtype=torch.float, device=y_generated.device) * data_label

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(y_generated, data_label_tensor)
    # ========================
    return loss

def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters

    dsc_optimizer.zero_grad()
    
    fake = gen_model.sample(x_data.shape[0], with_grad=False)
    
    real_output = dsc_model(x_data)
    fake_output = dsc_model(fake)
    
    dsc_loss = dsc_loss_fn(real_output, fake_output)
    dsc_loss.backward()
    dsc_optimizer.step()
    
    # ========================

    # Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    
    gen_optimizer.zero_grad()
    
    fake_output = dsc_model(gen_model.sample(x_data.shape[0], with_grad=False))
    gen_loss = gen_loss_fn(fake_output)
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
    
    if len(dsc_losses) % 1 == 0:
        saved = True
        torch.save(gen_model, checkpoint_file)

    return saved
