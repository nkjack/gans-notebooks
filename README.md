## Info

Implementation of different GANs.
To start off:
```
conda env create -f environment.yml
conda activate gans-notebooks
```

#### Datasets references
- Large Scale Scene Understanding (LSUN) (Yu et al. 2015)
- Imagenet-1k (Deng et al. 2009)
- MPIIGaze [Link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)
- UnityEyes [Link](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/)
- cycleGAN - [Link](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)


## Papers and Links

- GAN hacks - [git](https://github.com/soumith/ganhacks), [paper](https://arxiv.org/pdf/1606.03498.pdf)
- 2014
    - GAN - 10 June 2014 - [link](https://arxiv.org/pdf/1406.2661.pdf)
- 2016
    - Deep Convlutional GAN - 7 Jan 2016 - [link](https://arxiv.org/pdf/1511.06434.pdf)
    - Pixel RNN - 19 Aug 2016 - [link](https://arxiv.org/pdf/1601.06759.pdf)
- 2017
	- LSGAN - 5 Apr. 2017 - [link](https://arxiv.org/pdf/1611.04076.pdf)
	- Auxiliary Classifier GAN - 20 June 2017 [link](https://arxiv.org/pdf/1610.09585.pdf)
    - Sim GAN - 19 Jul 2017 - [link](https://arxiv.org/pdf/1612.07828.pdf)
    - StackGAN - Text to photorealistic image - 5 Aug 2017 - [link](https://arxiv.org/pdf/1612.03242.pdf), [v2](https://arxiv.org/pdf/1710.10916.pdf) 
        - I read the paper and tried to use pre-trained model unsuccessfully.
    - Wasserstein GAN - 6 Dec 2017 - [link](https://arxiv.org/pdf/1701.07875.pdf)
    - WGAN-gp - 25 Dec 2017 - [link](https://arxiv.org/pdf/1704.00028.pdf)
- 2018
    - Progressive growing GANs - 26 Feb 2018 [link](https://arxiv.org/pdf/1710.10196.pdf)
        - Need Tensorflow to run experiments from [git](https://github.com/tkarras/progressive_growing_of_gans).
    - Cycle GAN - 15 Nov. 2018 - [link](https://arxiv.org/pdf/1703.10593.pdf)
- 2019
    - sinGAN - ICCV 2019 - [link](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf)
    - Style GAN - 29 Mar. 2019 - [link](https://arxiv.org/pdf/1812.04948.pdf)
        - Need Tensorflow to run experiments.
    - SS-GAN (Self Supervised Gan) - 9 Apr 2019 - [link](https://arxiv.org/pdf/1811.11212.pdf), [git](https://github.com/vandit15/Self-Supervised-Gans-Pytorch)
        - I read it, it's nice, but there is no need to implement it.
    - Self attention GAN - 14 June 2019 - [link](https://arxiv.org/pdf/1805.08318.pdf)
- 2020
	- FreezeD - 28 Feb. 2020 - [link](https://arxiv.org/pdf/2002.10964.pdf)    



Notes from Implementation
-------------------------
#### DCGAN - Deep Convlutional GAN

Architecture guidelines for stable Deep Convolutional GANs:

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

Model:

- Scaling the range of the tanh activation funciton [-1, 1].
- All models trained with mini-batch (128) SGD.
- Initialization - zero-centered Normal distribution with standard deviation 0.02.
- LeakyReLU - slope - 0.2
- Using ADAM - change momentum from default 0.9 to 0.5

#### AC GAN
In the AC-GAN, every generated sample has a corresponding class c in addition to the noise z. G uses both to generate fake image. X_fake = G(z,c).

- They trained 100 AC-GAN moedls - each on images from just 10 classes - for 50,000 mini-batches of size 100.

#### Sim GAN

Using these datasets:
- MPIIGaze [Link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)
- UnityEyes [Link](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/)

#### Cycle GAN
[pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
[Torch](https://github.com/junyanz/CycleGAN)
[website](https://junyanz.github.io/CycleGAN/)

Real time style transfer Johnson et. al [Link](https://arxiv.org/pdf/1603.08155.pdf) [repo](https://github.com/jcjohnson/fast-neural-style)
Pix2Pix [Link](https://arxiv.org/pdf/1611.07004.pdf)

#### SAGAN (Self Attention GAN)
[code](https://github.com/brain-research/self-attention-gan)


#### StackGAN
Their code:

- orig - [tf](https://github.com/hanzhanggit/StackGAN), [pytorch](https://github.com/hanzhanggit/StackGAN-Pytorch)
- v2 - [pytorch](https://github.com/hanzhanggit/StackGAN-v2)

They propose:

- Novel Stacked Generative Adversarial Networks for synthesizing photo-realistic images from text descriptions. 
- A new Conditioning Augmentation technique is proposed to stabilize the conditional GAN training and also improves the diversity of the generated samples.
-  Extensive qualitative and quantitative experiments demonstrate the effectiveness of the overall model design as well as the effects of individual components, which provide useful information for designing future conditional GAN models.

#### WGAN
WGAN-gp - [git](https://github.com/igul222/improved_wgan_training)


#### Progressive growing GANs
[code](https://github.com/tkarras/progressive_growing_of_gans)
[trained models](https://drive.google.com/drive/folders/0B4qLcYyJmiz0NHFULTdYc05lX0U)

"As the training advances, we incrementally add layers to G and D, thus increasing the spatial resolution of the generated images. All existing layers remain trainable throughout the process."


#### Style GAN
[code](https://github.com/NVlabs/stylegan)


#### Extra Repo
https://github.com/google/compare_gan