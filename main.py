import argparse
import os

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
parser.add_argument('--draw_length', default=None, type=int)
parser.add_argument('--draw_dims', default=None, type=str)
parser.add_argument('--draw_diff', action='store_true')
parser.add_argument('-a', '--anomaly_percentage', default=0.04, type=float)
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
draw_diff = args.draw_diff
if args.draw_dims is not None:
    draw_dims = args.draw_dims.strip().split('.')
else:
    draw_dims = None
anomaly_percentage = args.anomaly_percentage

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device

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
if iters * batch_size != test_set_size:
    batch_x = test_set[iters * batch_size:]
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
recon_list = recon_list[1:]

print(type(recon_list), type(test_set))

mse_list = np.mean(np.square(recon_list - test_set.numpy()), axis=1)
print(mse_list.shape, recon_list.shape)
mse_sort_list = np.argsort(mse_list)[::-1]
# mse_sort_list=np.argsort(mse_list)
predicted_anomaly_position = mse_sort_list[:int(anomaly_percentage * test_set_size)]
true_anomaly_position = dataloader.load_anomaly_position()

plt.figure(dpi=300, figsize=(10, 5))
x = np.arange(mse_list.shape[0])
# y_predicted = np.zeros(mse_list.shape[0], dtype=float)
# y_gt = np.zeros(mse_list.shape[0], dtype=float)
if draw_length is None:
    x_predicted = predicted_anomaly_position
    x_gt = true_anomaly_position
else:
    x_predicted = predicted_anomaly_position[np.where(predicted_anomaly_position < draw_length)]
    x_gt = true_anomaly_position[np.where(true_anomaly_position < draw_length)]
y_predicted = mse_list[x_predicted]
y_gt = mse_list[x_gt]

plt.plot(x[:draw_length], mse_list[:draw_length])
# y1 = np.zeros(mse_list.shape[0]) + np.max(mse_list[:draw_length]) / 3
# y2 = np.zeros(mse_list.shape[0]) + 2 * np.max(mse_list[:draw_length]) / 3

# plt.scatter(x_gt, y_gt, label='gt', color='red', alpha=0.5,s=5)
for i in x_gt:
    plt.axvline(i, color='red', alpha=0.5)
plt.scatter(x_predicted, y_predicted, label='predicted', color='green', alpha=0.9, s=5)
plt.legend()
plt.savefig('plt.png', format='png')
plt.close()


# print(type(recon_list-test_set.numpy()))

def cal_metrics(gt, predicted, total):
    gt_oz = np.zeros(total, dtype=float)
    gt_oz[gt] += 1.
    pred_oz = np.zeros(total, dtype=float)
    pred_oz[predicted] += 1.
    tp = np.where((pred_oz == 1) & (gt_oz == 1), 1., 0.).sum()
    fp = np.where((pred_oz == 1) & (gt_oz == 0), 1., 0.).sum()
    tn = np.where((pred_oz == 0) & (gt_oz == 0), 1., 0.).sum()
    fn = np.where((pred_oz == 0) & (gt_oz == 1), 1., 0.).sum()
    # tplist = np.where((pred_oz == 1) & (gt_oz == 1), 1., 0.)
    # fplist = np.where((pred_oz == 1) & (gt_oz == 0), 1., 0.)
    # tnlist = np.where((pred_oz == 0) & (gt_oz == 0), 1., 0.)
    # fnlist = np.where((pred_oz == 0) & (gt_oz == 1), 1., 0.)
    # f = open('hel.txt', 'w')
    # print('pre', 'gt', 'tp', 'fp', 'tn', 'fn', sep='\t', file=f)
    # for i in range(tplist.shape[0]):
    #     print(pred_oz[i], gt_oz[i], tplist[i], fplist[i], tnlist[i], fnlist[i], sep='\t', file=f)
    # f.close()
    print(tp, fp, tn, fn)
    print(gt_oz.sum(), pred_oz.sum())
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    f1=2*precision*recall/(precision+recall)
    print(recall,precision,f1)


cal_metrics(true_anomaly_position, predicted_anomaly_position, mse_list.shape[0])

# draw

# print(recon_list.shape)
length = draw_length
if draw_dims is not None:
    for dim_str in draw_dims:
        dim = int(dim_str)
        if which_model == 'lstmvae':
            y1 = test_set[:length, -1, dim]
        else:
            y1 = test_set[:length, dim]
        y2 = recon_list[:length, dim]
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
        plt.savefig(os.path.join('save', dim_str + '.png'), format='png')
        print("plt")
