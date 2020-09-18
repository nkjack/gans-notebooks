from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from spectral import SpectralNorm

from torch.autograd import Variable

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
            # 1x28x28
            SpectralNorm(nn.Conv2d(in_channels, 64, kernel_size=4 , stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x14x14
            SpectralNorm(nn.Conv2d(64, 128, kernel_size=4 , stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x7x7
            SpectralNorm(nn.Conv2d(128, 256, kernel_size=4 , stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x3x3
#             nn.Conv2d(256, 1, kernel_size=4 , stride=2, padding=1, bias=False)
        ]
        
        self.feature_extractor = nn.Sequential(*first_modules)
        self.fc_gan = nn.Linear(256*3*3, 1)
        self.fc_rot = nn.Linear(256*3*3, 4)
        self.softmax = nn.Softmax(dim=-1)

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
        out = self.fc_gan(features.view(b_size, -1))
        rot_out = self.fc_rot(features.view(b_size, -1))
        return out, self.softmax(rot_out)


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim # = 100 or 1024
        
        all_modules = [
            SpectralNorm(nn.ConvTranspose2d(z_dim, 512, 3, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SpectralNorm(nn.ConvTranspose2d(512, 256, 3 ,stride=2, padding=0, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SpectralNorm(nn.ConvTranspose2d(256, 128, 4 ,stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4 ,stride=2, padding=1, bias=False),
            nn.Tanh()
        ]
        
        self.gen = nn.Sequential(*all_modules)
        

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
        # b_size = z.size(0)
        z = z.view(z.size(0), z.size(1), 1, 1)
        
        #output = self.fc(z)
        #x = self.gen(output.view(b_size, 256, 7, 7))
        x = self.gen(z)
        return x


def discriminator_loss_fn(y_data, y_generated, y_rot_logits ,data_label=0, batch_size=32, weight_d=1.0):
    assert data_label == 1 or data_label == 0
    device = y_data.device
    loss = torch.mean(((y_data - data_label)**2) +  ((y_generated - (1 - data_label))**2))
    
    rot_labels = torch.zeros(4*batch_size,).cuda()
    rot_labels[:batch_size] = 0
    rot_labels[batch_size:2*batch_size] = 1
    rot_labels[2*batch_size:3*batch_size] = 2
    rot_labels[3*batch_size:4*batch_size] = 3
    rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
    
    loss_rot = F.binary_cross_entropy_with_logits(input=y_rot_logits, target=rot_labels)
    loss += loss_rot*weight_d
    
    # ========================
    return loss


def generator_loss_fn(y_generated, y_rot_logits, data_label=0, batch_size=32, weight_g=0.5):
    assert data_label == 1 or data_label == 0

    device = y_generated.device
    loss = torch.mean((y_generated - data_label)**2)
    
    rot_labels = torch.zeros(4*batch_size,).cuda()
    rot_labels[:batch_size] = 0
    rot_labels[batch_size:2*batch_size] = 1
    rot_labels[2*batch_size:3*batch_size] = 2
    rot_labels[3*batch_size:4*batch_size] = 3
    rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
    
    loss_rot = F.binary_cross_entropy_with_logits(input=y_rot_logits, target=rot_labels)
    loss += loss_rot*weight_g
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
    fake = gen_model.sample(x_data.shape[0], with_grad=True)
    x_90 = fake.transpose(2,3)
    x_180 = fake.flip(2,3)
    x_270 = fake.transpose(2,3).flip(2,3)
    x_generated = torch.cat([x_data,x_90,x_180,x_270], 0)
    
    x_90 = x_data.transpose(2,3)
    x_180 = x_data.flip(2,3)
    x_270 = x_data.transpose(2,3).flip(2,3)
    x = torch.cat([x_data,x_90,x_180,x_270], 0)

    dsc_optimizer.zero_grad()
    
    real_output, real_logits = dsc_model(x) #.view(-1)
    fake_output, fake_logits = dsc_model(x_generated.detach())
    
    dsc_loss = dsc_loss_fn(real_output, fake_output, real_logits)
    dsc_loss.backward()
    dsc_optimizer.step()
    
    # ========================

    # Generator update 
    gen_optimizer.zero_grad()
    
    fake_output, fake_logits = dsc_model(x_generated)
    gen_loss = gen_loss_fn(fake_output, fake_logits)
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
