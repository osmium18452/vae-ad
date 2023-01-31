import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataLoader import DataLoader
from VAE import VAE
import matplotlib.pyplot as plt
from tqdm import tqdm
from AE import AE
from LSTMVAE import LSTMVAE

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_train_samples", default=-1, type=int)
parser.add_argument("-s", "--save_name", default="result.pkl", type=str)
parser.add_argument("-p", "--parent", default=5, type=int)
parser.add_argument("-v", "--variates", default=None, type=int)
parser.add_argument("--show", action="store_true")
parser.add_argument("--latent", default=5, type=int)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("-r", "--learning_rate", default=0.001, type=float)
parser.add_argument("-e", "--epoch", default=10, type=int)
parser.add_argument("-b", "--batch", default=1024, type=int)
parser.add_argument("-m", "--multivariate", action="store_true")
parser.add_argument("-w", "--window_size", default=20, type=int)
parser.add_argument("-g", "--gpu_device", default="0", type=str)
parser.add_argument('--cnn_lr', default=0.0001, type=float)
parser.add_argument('--vae_lr', default=0.001, type=float)
parser.add_argument('-N', '--normalize_data', action='store_true')
parser.add_argument('--model', default='vae', type=str)
parser.add_argument('--kl_weight', default=1., type=float)
parser.add_argument('--draw_length', default=100, type=int)
parser.add_argument('--draw_dim', default=0, type=int)
parser.add_argument('--draw_diff',action='store_true')
args = parser.parse_args()

num_train_samples = args.num_train_samples
save_name = args.save_name
parent = args.parent
variates = args.variates
show = args.show
latent = args.latent
gpu = args.gpu
learning_rate = args.learning_rate
total_epoch = args.epoch
batch_size = args.batch
univariate = not args.multivariate
multivariate = args.multivariate
window_size = args.window_size
gpu_device = args.gpu_device
cnn_lr = args.cnn_lr
vae_lr = args.vae_lr
normalize_data = args.normalize_data
which_model = args.model
kl_weight = args.kl_weight
draw_length = args.draw_length
draw_dim = args.draw_dim
draw_diff=args.draw_diff

train_file = 'machine-2-1.train.pkl'
test_file = 'machine-2-1.test.pkl'
# train_file = 'sector1.pkl'
# test_file = 'sector1.pkl'
label_file = 'machine-2-1.label.pkl'
graph_file = 'machine-2-1.camap.pkl'
dataloader = DataLoader(train_file, test_file, label_file, normalize=True, n_variate=variates)
dataloader.prepare_vae_data_set(T=window_size)
train_set = torch.Tensor(np.squeeze(dataloader.load_vae_train_set()))
test_set = torch.Tensor(np.squeeze(dataloader.load_vae_test_set()))
train_set_size = train_set.shape[0]
test_set_size = test_set.shape[0]

print(train_set.shape, test_set.shape)
input_size = train_set.shape[-1]
latent_size = latent
if which_model == 'vae':
    model = VAE(input_size, latent_size)
elif which_model == 'ae':
    model = AE(input_size, latent_size)
elif which_model == 'lstmvae':
    model = LSTMVAE(input_size, latent_size, window_size)
else:
    model = None

print(model)
if gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), learning_rate)
zeros = np.zeros((train_set.shape[-1], batch_size))

for epoch in range(total_epoch):
    if epoch % 10 == 0:
        permutation = np.random.permutation(train_set.shape[0])
        train_set = train_set[permutation]

    iters = train_set_size // batch_size
    with tqdm(total=iters, ascii=True) as pbar:
        pbar.set_postfix_str("epochs: --- train loss: -.-----e--- mse loss: -.-----e---, kl loss: -.-----e---")
        for i in range(iters):
            batch_x = train_set[i * batch_size:(i + 1) * batch_size]
            zero_loss = np.square(batch_x.numpy()).sum(axis=-1).mean()
            # print(batch_x)
            if gpu:
                batch_x = batch_x.cuda()

            if which_model == 'vae':
                recon, mu, log_std = model(batch_x)
                loss, recon_loss, kl_loss = model.loss_function(recon, batch_x, mu, log_std, kl_weight=kl_weight)
                pbar.set_postfix_str("epochs: %d/%d train loss: %.5e mse loss: %.5e, kl loss:%.5e" % (
                    epoch + 1, total_epoch, loss.item(), recon_loss.item(), kl_loss.item()))
            elif which_model == 'ae':
                recon = model(batch_x)
                loss = model.loss_function(batch_x, recon)
                recon_loss = F.mse_loss(recon, batch_x)
                pbar.set_postfix_str("epochs: %d/%d train loss: %.5e mse loss: %.5e, kl loss:%.5e" % (
                    epoch + 1, total_epoch, loss.item(), recon_loss.item(), loss.item() - recon_loss.item()))
            elif which_model == 'lstmvae':
                recon, mu, log_var = model(batch_x)
                loss, mse_loss, kl_loss = model.loss_function(recon, batch_x, mu, log_var)
                pbar.set_postfix_str("epochs: %d/%d train loss: %.5e mse loss: %.5e, kl loss:%.5e" % (
                    epoch + 1, total_epoch, loss.item(), mse_loss.item(), kl_loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()

        if iters * batch_size != train_set_size:
            batch_x = train_set[iters * batch_size:]
            if gpu:
                batch_x = batch_x.cuda()
            if which_model == 'vae':
                recon, mu, log_std = model(batch_x)
                loss, recon_loss, kl_loss = model.loss_function(recon, batch_x, mu, log_std)
                post_mse1 = torch.mean((recon[0] - batch_x[0]) ** 2)
                post_mse2 = torch.mean((recon - batch_x) ** 2)
            elif which_model == 'ae':
                recon = model(batch_x)
                loss = model.loss_function(batch_x, recon)
                # print(recon[0, :10], batch_x[0, :10], sep='\n')
            elif which_model == 'lstmvae':
                recon, mu, log_var = model(batch_x)
                loss, mse_loss, kl_loss = model.loss_function(recon, batch_x, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(recon[0],batch_x[0])

# exit()
iters = test_set_size // batch_size
recon_list = np.zeros(test_set.shape[-1]).reshape((1, -1))
with tqdm(total=iters, ascii=True) as pbar:
    for i in range(iters):
        batch_x = test_set[i * batch_size:(i + 1) * batch_size]
        # print(batch_x)
        if gpu:
            batch_x = batch_x.cuda()
        if which_model == 'vae':
            recon, mu, log_std = model(batch_x)
            recon_list = np.concatenate((recon_list, recon.cpu().detach().numpy()), axis=0)
        elif which_model == 'ae':
            recon = model(batch_x)
            recon_list = np.concatenate((recon_list, recon.cpu().detach().numpy()), axis=0)
        elif which_model == 'lstmvae':
            recon, mu, log_var = model(batch_x)
            recon_list = np.concatenate((recon_list, recon.cpu().detach().numpy()[:, -1]), axis=0)
        pbar.update()
recon_list = recon_list[1:]

print(recon_list.shape)
length = draw_length
if which_model == 'lstmvae':
    y1 = test_set[:length, -1, draw_dim]
else:
    y1 = test_set[:length, draw_dim]
y2 = recon_list[:length, draw_dim]
x = np.arange(y1.shape[0])
diff = np.abs(y1 - y2)
if which_model == 'lstmvae':
    # print(type((test_set[:length, -1] - recon_list[:length]) ** 2))
    mse = torch.mean((test_set[:length, -1] - recon_list[:length]) ** 2, dim=1)
else:
    mse = torch.mean((test_set[:length] - recon_list[:length]) ** 2, dim=1)
plt.figure(dpi=300, figsize=(10, 5))
ap = .9
plt.plot(x, y1, label='ground truth', alpha=ap)
plt.plot(x, y2, label='predicted', alpha=ap)
if draw_diff:
    plt.plot(x, diff, label='diff', alpha=ap)
# plt.plot(x, mse, label='mse', alpha=ap)
plt.legend()
plt.savefig('plot.png', format='png')
print("plt")
