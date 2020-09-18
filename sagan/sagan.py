from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from spectral import SpectralNorm

from torch.autograd import Variable

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8 , kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    
    def forward(self,x):
        """
        :param x: input feature maps(B X C X W X H)
        :returns out: self attention value + input feature 
        :returns attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        N = W*H
        proj_query  = self.query_conv(x).view(B, -1, N).permute(0,2,1) # B X N X C we don't want to affect the Batch dimension
        proj_key =  self.key_conv(x).view(B, -1, N) # B X C x N
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B X N X N
        proj_value = self.value_conv(x).view(B, -1, N) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B, C, W, H)
        
        out = self.gamma*out + x
        return out, attention

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
        ]
        
        self.attn = SelfAttention(256)
        self.last = nn.Conv2d(256, 1, kernel_size=4 , stride=2, padding=1, bias=False)
        
        self.feature_extractor = nn.Sequential(*first_modules)
        
        # h, w = self.in_size[1:]
        # h_gag, w_gag = h // (2 ** 4), w // (2 ** 4)
#         num_units = self.feature_extractor(torch.randn([1,3,153,153])).flatten().size(0) 
        
        # second_modules = [nn.Linear(h_gag * w_gag * 256, 1)]
#         second_modules = [nn.Linear(num_units, 1)]
        
#         self.features_to_score = nn.Sequential(*second_modules)
        
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
        out, p = self.attn(features)
        out = self.last(out)
        # ========================
        return out, p


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
            nn.ReLU()
        ]
        
        self.attn = SelfAttention(128)
        
        last = [
            nn.ConvTranspose2d(128, 1, 4 ,stride=2, padding=1, bias=False),
            nn.Tanh()
        ]

        self.gen = nn.Sequential(*all_modules)
        self.last = nn.Sequential(*last)
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
            samples, p = self.forward(z)
        else:
            with torch.no_grad():
                samples, p = self.forward(z)
        # ========================
        return samples, p

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
        out, p = self.attn(x)
        out = self.last(out)
        return out, p


def discriminator_loss_fn(y_data, y_generated, data_label=0):
    assert data_label == 1 or data_label == 0
    device = y_data.device
    loss = torch.mean(((y_data - data_label)**2) +  ((y_generated - (1 - data_label))**2))
    return loss


def generator_loss_fn(y_generated, data_label=0):
    assert data_label == 1 or data_label == 0
    device = y_generated.device
    loss = torch.mean((y_generated - data_label)**2)
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
    dsc_optimizer.zero_grad()
    
    fake, _ = gen_model.sample(x_data.shape[0], with_grad=True)
    
    real_output, _ = dsc_model(x_data)#.view(-1)
    fake_output, _ = dsc_model(fake.detach())
    
    dsc_loss = dsc_loss_fn(real_output, fake_output)
    dsc_loss.backward()
    dsc_optimizer.step()
    
    # ========================

    # Generator update    
    gen_optimizer.zero_grad()
    
    fake_output, _ = dsc_model(fake)
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
    
    if len(dsc_losses) % 5 == 0:
        saved = True
        torch.save(gen_model, checkpoint_file)

    return saved
