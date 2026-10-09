"""
mostly code from

https://github.com/fabiopapais/sparse-interpretability/blob/main/autoencoder.py

SPARSE AUTOENCODERS FIND HIGHLY INTER-
PRETABLE FEATURES IN LANGUAGE MODELS
"""

import torch
from torch import nn
from torch.nn import functional as F


class TiedAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(TiedAutoencoder, self).__init__()

        self.weights = nn.Parameter(torch.randn(input_dim, latent_dim))

        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        z = F.linear(x, self.weights.T)
        z = F.relu(z, inplace=True)
        reconstruction = F.linear(z, self.weights)

        return reconstruction, z
