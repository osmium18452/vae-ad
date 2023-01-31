import torch
from torch import nn
import torch.nn.functional as F


class LSTMVAE(nn.Module):
    def __init__(self, input_size, latent_size, window_size):
        super(LSTMVAE, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_size_list = [70, ]

        self.en_lstm_1 = nn.LSTM(input_size, self.hidden_size_list[0], batch_first=True)
        self.de_lstm_1 = nn.LSTM(latent_size, self.hidden_size_list[-1], batch_first=True)

        self.mean = nn.Linear(self.hidden_size_list[-1] * window_size, latent_size * window_size)
        self.log_var = nn.Linear(self.hidden_size_list[-1] * window_size, latent_size * window_size)

        self.output = nn.Linear(self.hidden_size_list[-1] * window_size, input_size * window_size)
        self.output2 = nn.Linear(self.hidden_size_list[-1], input_size)

    def encode(self, x):
        out, (_, __) = self.en_lstm_1(x)
        out = F.relu(out)
        out = torch.flatten(out, start_dim=1)
        mu = F.relu(self.mean(out))
        log_var = F.relu(self.log_var(out))
        return mu, log_var

    def decode(self, z):
        out, (_, __) = self.de_lstm_1(z)
        out = F.relu(out)
        out = torch.flatten(out, start_dim=1)
        # out = F.tanh(self.output(out))
        out = F.elu(self.output(out))
        # out = self.output(out)
        out = torch.reshape(out, (-1, self.window_size, self.input_size))
        return out

    def decode2(self, z):
        _, (out, __) = self.de_lstm_1(z)
        out = self.output2(F.relu(out))
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        out = eps * std + mu
        out = torch.reshape(out, (-1, self.window_size, self.latent_size))
        return out

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var

    def loss_function(self, recon, ori, mu, log_var):
        mse_loss = F.mse_loss(recon[:,-1], ori[:,-1], reduction='sum')
        kl_loss = -0.5 * (1 + 2 * log_var - mu.pow(2) - torch.exp(2 * log_var))
        kl_loss = torch.sum(kl_loss)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = mse_loss + kl_loss
        return loss, F.mse_loss(recon[:,-1], ori[:,-1]), kl_loss
