import torch
import math
from torch import nn
from typing import List

""" 
Inspiration from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py 
"""


def clean_layers(layers: List, threeD: bool):
    """
    :param threeD: bool indicating whether or not the layer format is for a 3d network
    :type layers: List
    :return layers: List with all instances of None converted to tuples of 0s.
    """
    if threeD:
        for module in layers:
            if module[6] is None:
                module[6] = (0, 0, 0)
            if module[7] is None:
                module[7] = (0, 0, 0)
        return layers

    for module in layers:
        if module[6] is None:
            module[6] = (0, 0)
        if module[7] is None:
            module[7] = (0, 0)
    return layers


def get_result_dim(img_dim: int, kernel_size: int, stride: int, padding: int):
    """ Assumes that the image in question is square """
    assert kernel_size > 0, "kernel_size is less than or equal to 0"
    assert img_dim > 0, "img_dimension is less than or equal to 0"
    assert stride > 0, "stride is less than or equal to 0"
    assert padding > -1, "padding is less than 0"
    return math.floor(((img_dim - kernel_size + 2 * padding) / stride) + 1)


class UnFlatten(nn.Module):
    def forward(self, x, in_dimension):
        return x.view(-1, self.layers[-1], 45, 45)


class ConvVAE(nn.Module):
    def __init__(self, img_dim: int, layers: List, in_channels: int, latent_dim: int, cuda: bool):
        super(ConvVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.layers = layers
        self.cuda = cuda

        # Encoder
        modules = []
        dim = img_dim
        for tuple in layers:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=tuple[0],
                              kernel_size=tuple[1],
                              stride=tuple[2],
                              padding=tuple[3]),
                    nn.ReLU()
                )
            )
            dim = get_result_dim(img_dim=dim, kernel_size=tuple[1], stride=tuple[2], padding=tuple[3])
            in_channels = tuple[0]

        modules.append(
            nn.Sequential(
                nn.Flatten()
            )
        )

        self.encoder = nn.Sequential(*modules)
        self.mu_layer = nn.Linear(in_features=(layers[-1][0] * (dim ** 2)),
                                  out_features=self.latent_dim)
        self.logvar_layer = nn.Linear(in_features=(layers[-1][0] * (dim ** 2)),
                                      out_features=self.latent_dim)
        self.transition_layer = nn.Linear(in_features=self.latent_dim,
                                          out_features=layers[-1][0] * (dim ** 2))

        # Decoder
        modules = []
        layers.reverse()

        modules.append(
            nn.Sequential(
                UnFlatten()
            )
        )

        for i in range(len(layers) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=layers[i][0],
                                       out_channels=layers[i + 1][0],
                                       kernel_size=layers[i][1],
                                       stride=layers[i][2],
                                       padding=layers[i][3]),
                    nn.ReLU()
                )
            )

        # Final Layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=layers[-1][0],
                                   out_channels=self.in_channels,
                                   kernel_size=layers[-1][1],
                                   stride=layers[-1][2],
                                   padding=layers[-1][3]
                                   )
            )
        )

        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state):
        post_encoder = self.encoder(state)
        mu, logvar = self.mu_layer(post_encoder), self.logvar_layer(post_encoder)
        z = self.reparameterize(mu, logvar)
        z = self.transition_layer(z)
        post_decoder = self.decoder(z)
        return post_decoder, mu, logvar


class Conv3dVAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, channels: int, layers: List, latent_dim: int, device: torch.device):
        super(Conv3dVAE, self).__init__()
        """
        :param x_dim: Assuming input images are square, this defines the length of a side of the square.
        :param z_dim: The number of images in a block of input, the z dimension of the cuboid.
        :param channels: The number of input channels in each image.
        :param layers: List of lists of length n, where n is the number of layers you wish to implement in encoder and decoder.
                       Each tuple has the following structure:
                        (in_channels, out_channels, kernel_size_x, kernel_size_z, stride_x, stride_z, padding:tuple, output_padding:tuple).
                        
                        Note: The padding parameter should only be used for the forward pass - use the output_padding in the
                        backward pass. If you need to add padding to the beginning of a layer during the backward pass, 
                        change the value of the previous layer's output_padding property instead of using the padding
                        parameter.
        :param latent_dim: Integer describing the number of latent dimensions (size of the mu and logvar encoded representations)
        :param device: Torch.device object describing the device to train the model on.
        """
        new_layers = clean_layers(layers, True)

        self.device = device
        self.latent_dim = latent_dim
        self.layers = new_layers
        self.channels = channels
        self.z_dim = z_dim
        self.x_dim = x_dim

        modules = []
        current_channels = channels
        current_x_dim = x_dim
        current_z_dim = z_dim

        # Encoder
        for module in self.layers:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=module[0],
                              out_channels=module[1],
                              kernel_size=(module[3], module[2], module[2]),
                              stride=(module[5], module[4], module[4]),
                              padding=module[6]
                              ),
                    nn.ReLU()
                )
            )
            current_channels = module[1]

            if module[6] is None:
                module[6] = [0, 0, 0]

            current_x_dim = get_result_dim(current_x_dim, kernel_size=module[2], stride=module[4], padding=module[6][1])
            current_z_dim = get_result_dim(current_z_dim, kernel_size=module[3], stride=module[5], padding=module[6][0])

        self.encoder = nn.Sequential(*modules)

        # Latent dimensions
        self.mu_layer = nn.Conv3d(in_channels=current_channels, out_channels=latent_dim,
                                  kernel_size=(current_z_dim, current_x_dim, current_x_dim), stride=(1, 1, 1))
        self.logvar_layer = nn.Conv3d(in_channels=current_channels, out_channels=latent_dim,
                                      kernel_size=(current_z_dim, current_x_dim, current_x_dim), stride=(1, 1, 1))
        self.transition_layer = nn.ConvTranspose3d(in_channels=latent_dim, out_channels=current_channels,
                                                   kernel_size=(current_z_dim, current_x_dim, current_x_dim),
                                                   stride=(1, 1, 1))

        layers.reverse()
        modules = []

        # Decoder
        for i in range(len(self.layers) - 1):
            module = self.layers[i]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels=module[1],
                                       out_channels=module[0],
                                       kernel_size=(module[3], module[2], module[2]),
                                       stride=(module[5], module[4], module[4]),
                                       padding=module[6],
                                       output_padding=module[7]
                                       ),
                    nn.ReLU()
                )
            )

        # Final layer - no ReLU activation
        module = layers[-1]
        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=module[1],
                    out_channels=module[0],
                    kernel_size=(module[3], module[2], module[2]),
                    stride=(module[5], module[4], module[4]),
                    padding=(0, 0, 0),
                    output_padding=module[7]
                )
            )
        )

        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)
        z = self.transition_layer(z)
        y = self.decoder(z)

        if y.shape[1:] != (self.channels, self.x_dim, self.z_dim):
            y = y[:, :self.channels, :self.z_dim, :self.x_dim, :self.x_dim]

        probs = torch.sigmoid(y)
        return y, probs, mu, logvar