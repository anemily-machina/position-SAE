"""
mostly code from

https://github.com/fabiopapais/sparse-interpretability/blob/main/autoencoder.py

SPARSE AUTOENCODERS FIND HIGHLY INTER-
PRETABLE FEATURES IN LANGUAGE MODELS
"""

from dotenv import load_dotenv

load_dotenv()

from buffer import ActivationBuffer, make_subsampling_buffer
from trainers.tied import TiedAutoencoder
from utils import make_folder, set_random_seeds


import argparse
import os


import torch
from torch import nn
from torch.nn import functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TiedAutoencoder with sparse coding"
    )
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=1e-5)
    return parser.parse_args()


def train_autoencoder(
    buffer: ActivationBuffer,
    learning_rate=1e-3,
    num_epochs=100,
    hidden_size=30000,
    alpha=1e-5,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    first_batch = next(buffer)
    input_dim = first_batch.size(1)

    model = TiedAutoencoder(input_dim, hidden_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"input_dim={input_dim}, latent_dim={hidden_size}")

    losses = {
        "epoch_losses": [],
        "reconstruction_losses": [],
        "sparsity_losses": [],
    }

    epoch_size = 8192 * 32

    for epoch in range(num_epochs):

        epoch_reconstruction_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_loss = 0.0
        total_batches = 0

        end_epoch = epoch_size

        for batch in buffer:

            batch = batch.to(device)
            batch = batch.float()

            optimizer.zero_grad()
            reconstructed, latent = model(batch)

            reconstruction_loss = criterion(reconstructed, batch)
            sparsity_loss = alpha * torch.abs(latent).mean()

            # sparsity_loss = alpha * model.get_latent_sum()

            loss = reconstruction_loss + sparsity_loss

            loss.backward()
            optimizer.step()

            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_loss += loss.item()
            total_batches += 1

            end_epoch -= len(batch)

            if end_epoch <= 0:
                break

        avg_epoch_loss = epoch_loss / total_batches
        avg_epoch_reconstruction_loss = epoch_reconstruction_loss / total_batches
        avg_epoch_sparsity_loss = epoch_sparsity_loss / total_batches

        losses["epoch_losses"].append(avg_epoch_loss)
        losses["reconstruction_losses"].append(avg_epoch_reconstruction_loss)
        losses["sparsity_losses"].append(avg_epoch_sparsity_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} avg loss: {avg_epoch_loss:.6f}, avg recons. loss: {avg_epoch_reconstruction_loss:.6f}, avg sparsity loss: {avg_epoch_sparsity_loss:.6f}"
        )

    return model, losses


if __name__ == "__main__":

    set_random_seeds(4321)

    buffer = make_subsampling_buffer(0.2)

    args = parse_args()

    model, losses = train_autoencoder(
        buffer=buffer,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_size=args.hidden_size,
        alpha=args.alpha,
    )

    print(f"Final loss: {losses['epoch_losses'][-1]:.6f}")
    print(f"Final reconstruction loss: {losses['reconstruction_losses'][-1]:.6f}")
    print(f"Final sparsity loss: {losses['sparsity_losses'][-1]:.6f}")
