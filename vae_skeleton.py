import torch
from torch import nn
from typing import List

""" 
Inspiration from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py 

Intended behaviour:

layers = [(out_channels, kernel, stride, padding), (kernel2, stride2, padding2) ... ]

TODO: Test and integrate.
"""

def get_result_dim(img_dim: int, kernel_size: int, stride: int, padding: int):
    """ Assumes that the image in question is square """
    assert kernel_size > 0, "kernel_size is less than or equal to 0"
    assert img_dim > 0, "img_dimension is less than or equal to 0"
    assert stride > 0, "stride is less than or equal to 0"
    assert padding > -1, "padding is less than 0"
    return ((img_dim - kernel_size + 2 * padding) / stride) + 1


class UnFlatten(nn.Module):
    def forward(self, x, in_dimension):
        return x.view(-1, self.layers[-1], 45, 45)


class ConvVAE(nn.Module):
    def __init__(self, img_dim: int, layers: List, in_channels: int, latent_dim: int, cuda:bool):
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
                                       out_channels=layers[i+1][0],
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
