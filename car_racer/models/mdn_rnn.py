"""
M (Memory Model): Recurrent World Model (GRU + Mixture Density Network + Reward Head).
Predicts the distribution over the next latent code z and the scalar reward.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from car_racer.config import Config

class MDNRNN(nn.Module):
    def __init__(self,
                 latent_dim=Config.LATENT_DIM,
                 action_dim=Config.ENV_ACTIONS,
                 hidden_dim=Config.RNN_HIDDEN_DIM,
                 num_gaussians=Config.NUM_GAUSSIANS):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.k = num_gaussians

        # GRU input: [latent_z, action_one_hot]
        self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)

        # MDN heads
        self.pi = nn.Linear(hidden_dim, self.k)
        self.mu = nn.Linear(hidden_dim, self.k * latent_dim)
        self.logsig = nn.Linear(hidden_dim, self.k * latent_dim)

        # Learned reward head: predicts reward directly in the dream
        self.rhead = nn.Linear(hidden_dim, 1)

    def forward(self, z, a_onehot, h=None):
        """
        z: (B, T, latent_dim)
        a_onehot: (B, T, action_dim)
        h: GRU hidden state (1, B, hidden_dim) or None
        """
        x = torch.cat([z, a_onehot], dim=-1)
        out, h = self.rnn(x, h)
        B, T_, _ = out.shape

        pi = self.pi(out)
        mu = self.mu(out).view(B, T_, self.k, self.latent_dim)
        logsig = self.logsig(out).view(B, T_, self.k, self.latent_dim).clamp(-5, 2)
        rhat = self.rhead(out).squeeze(-1)

        return pi, mu, logsig, rhat, h

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=Config.DEVICE):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()

def mdn_loss(pi, mu, logsig, target):
    """
    Negative log-likelihood of target under Gaussian mixture.
    pi: (B, T, K)
    mu: (B, T, K, Z)
    logsig: (B, T, K, Z)
    target: (B, T, Z)
    """
    t = target.unsqueeze(2)  # (B, T, 1, Z)
    # log p(target | component k)
    logp = -0.5 * (((t - mu) / logsig.exp()) ** 2 + 2 * logsig + math.log(2 * math.pi))
    logp = logp.sum(-1) + F.log_softmax(pi, -1)  # sum over latent dimensions + log mixture weight
    return -torch.logsumexp(logp, -1).mean()

def sample_mdn(pi, mu, logsig, temperature=1.0, generator=None):
    """
    Sample candidate future z from the predicted Gaussian mixture distribution.
    pi: (B, K)
    mu: (B, K, Z)
    logsig: (B, K, Z)
    returns: (B, Z) sampled latent vectors
    """
    temperature = max(1e-4, float(temperature))
    probs = F.softmax(pi / temperature, -1)
    k = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    b_idx = torch.arange(len(k), device=pi.device)

    sel_mu = mu[b_idx, k]
    sel_sig = logsig[b_idx, k].exp() * math.sqrt(temperature)

    noise = torch.randn_like(sel_mu) if generator is None else torch.randn(sel_mu.shape, generator=generator, device=pi.device)
    return sel_mu + noise * sel_sig
