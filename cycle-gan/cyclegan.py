from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import numpy as np

## from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class Identity(nn.Module):
    def forward(self, x):
        return x

class ResnetGenerator(nn.Module):
    def __init__(self, use_dropout=False):
        """Construct a Resnet-based generator
        Parameters:
            use_dropout (bool)  -- if use dropout layers
            
            c7s1-64,d128,d256,R256,R256,R256,
            R256,R256,R256,R256,R256,R256,u128
            u64,c7s1-3
        """
        super(ResnetGenerator, self).__init__()

        n_blocks=9
        ngf=64
#         norm_layer=nn.BatchNorm2d
        # c7s1
        model = [
             nn.ReflectionPad2d(3),
             nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=True),
             nn.InstanceNorm2d(64),
             nn.ReLU(True)
        ]

        # d128, d256
        model += [
            nn.Conv2d(64 , 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            ##
            nn.Conv2d(128 , 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]
        
        # add ResNet blocks
        for i in range(n_blocks):       
            model += [
                ResnetBlock(256)
            ]

        # u128, u64
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]

        # c7s1-3
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        """Construct a convolutional block.
        Parameters:
            :dim: (int) the number of channels in the conv layer.
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(dim), 
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
    

class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            
        C64-C128-C256-C512
        """
        super(PatchGANDiscriminator, self).__init__()
        
        sequence = [
            # c64
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),  #padding 128?
            nn.LeakyReLU(0.2, True),
            # c128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # c256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # c512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # output 1 channel prediction map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        ]
    
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)    

    

def discriminator_loss_fn(y_data, y_generated, data_label=0):
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
    #  Pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    device = y_data.device
    b_size = y_data.size(0)

    loss_data = torch.mean(((y_data - data_label)**2) +  ((y_generated - (1 - data_label))**2))
    return loss_data


def generator_loss_fn(y_generated, data_label=0):
    """
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    
    loss = torch.mean((y_generated - data_label)**2)
    return loss

def cycle_consistency_loss_fn(x,xyx_generated, lam=10):
    loss_fn = nn.L1Loss()
    l1 = loss_fn(x, xyx_generated)
    
    return lam * l1

def idtentity_loss_fn(x, x_idt, lam_idt):
    loss_fn = nn.L1Loss()
    loss = loss_fn(x, x_idt) 
    return loss * lam_idt

def train_batch(dscY: PatchGANDiscriminator, dscX: PatchGANDiscriminator,
                genXY: ResnetGenerator, genYX: ResnetGenerator,
                dsc_loss_fn: Callable, 
                gen_loss_fn: Callable, 
                cyc_loss_fn: Callable, 
                idt_loss_fn:Callable, 
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_real: DataLoader, y_real: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
    
    # forward
    # compute fake images and reconstruction images.    
    fakeY = genXY(x_real)
    fakeX = genYX(y_real)
    fakeXYX = genYX(fakeY)
    fakeYXY = genXY(fakeX)
    
    # G_A and G_B
    # Ds require no gradients when optimizing Gs
    for param in dscX.parameters():
        param.requires_grad = False
    
    for param in dscY.parameters():
        param.requires_grad = False
    
    
    gen_optimizer.zero_grad()  # set GenXY and GenYX's gradients to zero
    
    # calculate gradients for GenXY and GenYX

    # Identity loss
    loss_idt_X = 0
    loss_idt_Y = 0
    
#     if lambda_idt > 0:
    idtX = genXY(x_real)
    loss_idt_X = idt_loss_fn(idtX, x_real) # * lambda_Y * lambda_idt

    idtY = genYX(y_real)
    loss_idt_Y = idt_loss_fn(idtY, y_real) # * lambda_X * lambda_idt
      

    # GAN loss D_Y(G_XY(X))
    loss_G_XY = gen_loss_fn(dscY(fakeY))
    # GAN loss D_X(G_YX(Y))
    loss_G_YX = gen_loss_fn(dscX(fakeX)) 
    # Forward cycle loss || G_YX(G_XY(X)) - X||
    loss_cycle_XYX = cyc_loss_fn(fakeXYX, x_real) #* lambda_X
    # Backward cycle loss || G_XY(G_YX(Y)) - Y||
    loss_cycle_YXY = cyc_loss_fn(fakeYXY, y_real) #* lambda_Y
    # combined loss and calculate gradients
    loss_G = loss_G_XY + loss_G_YX + loss_cycle_XYX + loss_cycle_YXY + loss_idt_X + loss_idt_Y
    loss_G.backward()
         
    gen_optimizer.step()       # update GenXY and GenYX
    
    # D_X and D_Y
    for param in dscX.parameters():
        param.requires_grad = True
    
    for param in dscY.parameters():
        param.requires_grad = True
        
    dsc_optimizer.zero_grad()   # set D_X and D_Y's gradients to zero
    
    
    # Dsc Y
    pred_realY = dscY(y_real)
    pred_fakeY = dscY(fakeY.detach())
    loss_DY = dsc_loss_fn(pred_realY, pred_fakeY)
    loss_DY.backward()
    
    # Dsc X
    pred_realX = dscX(x_real)
    pred_fakeX = dscX(fakeX.detach())
    loss_DX = dsc_loss_fn(pred_realX, pred_fakeX)
    loss_DX.backward()
    
    dsc_optimizer.step()  # update D_X and D_Y
    
    return loss_DX.item(), loss_DY.item(), loss_G.item()

def save_checkpoint(gen_model, dscX_losses, dscY_losses, gen_losses, checkpoint_file):
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
    
#     if len(dscX_losses) % 5 == 0:
    saved = True
    torch.save(gen_model, checkpoint_file)

    return saved
