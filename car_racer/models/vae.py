"""
V (Vision Model): Convolutional Variational Autoencoder (ConvVAE).
Compresses 24x24x3 frame (1,728 numbers) down to latent code z (16 numbers).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from car_racer.config import Config

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=Config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 3 x 24 x 24 -> 1152 features
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),   # 16 x 12 x 12
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # 32 x 6 x 6
            nn.Flatten()                                            # 1152
        )
        self.mu = nn.Linear(1152, latent_dim)
        self.logvar = nn.Linear(1152, latent_dim)

        # Decoder: latent_dim -> 3 x 24 x 24
        self.fc = nn.Linear(latent_dim, 1152)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        """
        x: (B, 24, 24, 3) normalized float image tensor
        returns: mu, logvar of shape (B, latent_dim)
        """
        # Convert channel last (B, H, W, C) to channel first (B, C, H, W)
        h = self.enc(x.permute(0, 3, 1, 2))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar, add_noise_std=0.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if add_noise_std > 0.0:
            z = z + add_noise_std * torch.randn_like(z)
        return z

    def decode(self, z):
        """
        z: (B, latent_dim)
        returns: reconstructed frame (B, 24, 24, 3)
        """
        h = self.fc(z).view(-1, 32, 6, 6)
        xhat = self.dec(h).permute(0, 2, 3, 1)
        return xhat

    def forward(self, x, add_noise_std=0.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, add_noise_std=add_noise_std)
        xhat = self.decode(z)
        return xhat, mu, logvar

    @staticmethod
    def vae_loss(x, xhat, mu, logvar, track_tensor, car_weight=Config.VAE_CAR_WEIGHT):
        """
        Reconstruction loss weighted heavily where pixels differ from the empty track
        + Analytical KL Divergence.
        """
        # Weight what moves (car is ~4 pixels out of 576)
        w = 1.0 + car_weight * (x - track_tensor).abs().mean(-1, keepdim=True)
        recon = (w * F.binary_cross_entropy(xhat, x, reduction="none")).sum() / len(x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
        return recon + 1.0 * kl, recon, kl

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=Config.DEVICE):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()
