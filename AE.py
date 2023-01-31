import torch
from torch import nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self,input_size, latent_size):
        super(AE, self).__init__()
        encoder_hidden_size=[25,10,5]
        decoder_hidden_size=[5,10,25]
        self.encoder=nn.Sequential(
            nn.Linear(input_size,encoder_hidden_size[0]),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(encoder_hidden_size[0],encoder_hidden_size[1]),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(encoder_hidden_size[1],encoder_hidden_size[2]),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(encoder_hidden_size[2],latent_size),
            nn.ReLU(),
            # nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, decoder_hidden_size[0]),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(decoder_hidden_size[0], decoder_hidden_size[1]),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(decoder_hidden_size[1], decoder_hidden_size[2]),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(decoder_hidden_size[2], input_size),
            # nn.Tanh(),
        )

    def forward(self,x):
        encoder=self.encoder(x)
        decoder=self.decoder(encoder)
        return decoder

    def loss_function(self,input,recon):
        return F.mse_loss(input,recon)
