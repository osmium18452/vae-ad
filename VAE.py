import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, input_size, latent_size, label_size=0):
        super(VAE, self).__init__()

        # neuron_list = [30, 20, 10]
        neuron_list = [70, 35, 15]

        self.encoder = nn.Sequential(
            nn.Linear(input_size, neuron_list[0]),
            nn.BatchNorm1d(neuron_list[0]),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(neuron_list[0], neuron_list[1]),
            nn.BatchNorm1d(neuron_list[1]),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(neuron_list[1], neuron_list[2]),
            nn.BatchNorm1d(neuron_list[2]),
            # nn.ReLU(),
            nn.Tanh(),
        )
        self.encoder_mean = nn.Linear(neuron_list[2], latent_size)
        self.encoder_log_std = nn.Linear(neuron_list[2], latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, neuron_list[2]),
            nn.BatchNorm1d(neuron_list[2]),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(neuron_list[2], neuron_list[1]),
            nn.BatchNorm1d(neuron_list[1]),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(neuron_list[1], neuron_list[0]),
            nn.BatchNorm1d(neuron_list[0]),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(neuron_list[0], input_size),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )

        # self.fc1 = nn.Linear(input_size + label_size, 200)
        # self.fc2_mu = nn.Linear(200, latent_size)
        # self.fc2_log_std = nn.Linear(200, latent_size)
        # self.fc3 = nn.Linear(latent_size + label_size, 200)
        # self.fc4 = nn.Linear(200, input_size)

    def encode(self, x):
        # print("encoder",x.shape,x.dtype)
        h1 = self.encoder(x)
        mean = self.encoder_mean(h1)
        log_std = self.encoder_log_std(h1)
        return mean, log_std

        # h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))  # concat features and labels
        # mu = self.fc2_mu(h1)
        # log_std = self.fc2_log_std(h1)
        # return mu, log_std

    def decode(self, z):
        recon = self.decoder(z)
        return recon

        # h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))  # concat latents and labels
        # recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std,kl_weight=1.) :
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 *(1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_weight*kl_loss
        loss_mse=F.mse_loss(recon,x)
        return loss,loss_mse,kl_loss
