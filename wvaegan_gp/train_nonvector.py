###----------------------------------------------------------------------------------
### Encoderにlabel情報を与えないverのモデル定義（GPU用）
###------------------------------------------------------------------------------------
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
import statistics
from wvaegan_gp.model_nonvector import Encoder, Decoder, Discriminator
from vaegan.util import save_loss, to_cpu, save_coords, to_cuda


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate") # 1e-4
parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient") # 0.0
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient") # 0.9
parser.add_argument("--latent_dim", type=int, default=6, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=496, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--sample_interval", type=int, default=10000, help="interval betwen image samples")
opt = parser.parse_args()

coord_shape = (opt.channels, opt.coord_size)

cuda = True if torch.cuda.is_available() else False
#cuda = False
lambda_gp = 10

# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Loss weight for gradient penalty
done_epoch = 0
if done_epoch>0:
    E_PATH = "wvaegan_gp/results/encoder_params_{0}".format(done_epoch)
    G_PATH = "wvaegan_gp/results/decoder_params_{0}".format(done_epoch)
    D_PATH = "wvaegan_gp/results/discriminator_params_{0}".format(done_epoch)
    encoder = Encoder(opt.latent_dim)
    encoder.load_state_dict(torch.load(E_PATH, map_location=torch.device('cpu')))
    encoder.eval()
    decoder = Decoder(opt.latent_dim)
    decoder.load_state_dict(torch.load(G_PATH, map_location=torch.device('cpu')))
    decoder.eval()
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(D_PATH, map_location=torch.device('cpu')))
    discriminator.eval()
else:
    encoder = Encoder(opt.latent_dim)
    decoder = Decoder(opt.latent_dim)
    discriminator = Discriminator()

if cuda:
    print("use GPU")
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()

# Configure data loader
perfs_npz = np.load("dataset/standardized_upsampling_perfs.npz")
coords_npz = np.load("dataset/standardized_upsampling_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]
perfs = perfs_npz[perfs_npz.files[0]]
perf_mean = perfs_npz[perfs_npz.files[1]]
perf_std = perfs_npz[perfs_npz.files[2]]

max_cl = 2.00
print(perfs.shape, coords.shape)
plt.hist(perfs)

dataset = torch.utils.data.TensorDataset(torch.tensor(coords), torch.tensor(perfs))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_image(epoch=None, data_num=12):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, opt.latent_dim))))
    labels = max_cl*np.random.random_sample(size=(data_num, opt.n_classes))
    labels = Variable(FloatTensor(labels))
    en_coords = to_cpu(decoder(z, labels)).detach().numpy()
    labels = to_cpu(labels).detach().numpy()
    if epoch is not None:
        save_coords(en_coords*coord_std+coord_mean, labels, "wvaegan_gp/coords/epoch_{0}_dim{1}".format(str(epoch).zfill(3), opt.latent_dim))
    else:
        np.savez("wvaegan_gp/results/final_dim{0}".format(opt.latent_dim), labels, en_coords*coord_std+coord_mean)
        save_coords(en_coords*coord_std+coord_mean, labels, "wvaegan_gp/coords/final_dim{0}.png".format(opt.latent_dim))

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for VAEGAN"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Training
# ----------
"""
start = time.time()
enc_losses, dec_losses, dis_losses = [], [], []
batches_done = 0
for epoch in range(opt.n_epochs-done_epoch):
    epoch+=done_epoch
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]
        #coords = coords.reshape(batch_size, *coord_shape)
        
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=True)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=True)
        
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_dis.zero_grad()
        
        # Configure input
        real_imgs = Variable(coords.type(FloatTensor))
        labels = to_cuda(Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes))))
        
        #ランダムノイズzと指定ラベルをdecoderに与える
        Z_p = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # Generate a batch of images
        X_gen = decoder(Z_p, labels)
        
        # Loss for real images
        validity_disc_real = discriminator(real_imgs, labels)
        
        # Loss for fake images
        validity_disc_fake = discriminator(X_gen, labels)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.reshape(real_imgs.shape[0], *coord_shape).data, X_gen.data, labels)
        dis_loss = -torch.mean(validity_disc_real) + torch.mean(validity_disc_fake) + lambda_gp * gradient_penalty
        dis_loss.backward()
        optimizer_dis.step()
        
        
        
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        
        # -----------------------------------------------------
        # Train encoder and decoder every n_critic iterations
        # -----------------------------------------------------
        if i % opt.n_critic == 0:
            # ---------------------------
            #  Train Encoder and Decoder
            # --------------------------
            #データセットをencode
            mus, log_variances = encoder(real_imgs, opt.latent_dim)
            variances = torch.exp(log_variances * 0.5)
            
            #修正箇所12/28,ここから
            Z_p = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            #ここまで
            
            Z = Z_p * variances + mus
            
            #Zをdecodeしてreconstructed_imgを取得
            X_recon = decoder(Z, labels)
            
            kl_div = -0.5*torch.sum(-log_variances.exp() - torch.pow(mus, 2) + log_variances +1)
            latent_loss = torch.sum(kl_div).clone().detach().requires_grad_(True)
            validity_real = discriminator(real_imgs, labels) # Loss for real images
            validity_fake = discriminator(X_recon, labels) # Loss for fake images
            discrim_layer_recon_loss = torch.mean(torch.square(validity_real - validity_fake)).clone().detach().requires_grad_(True)
            
            
            # Sample noise and labels as decoder input
            Z_p = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            labels_p = Variable(FloatTensor(max_cl*np.random.random_sample(size=(batch_size, opt.n_classes))))
            # Generate a batch of images
            X_p = decoder(Z_p, labels_p)
        
            validity_gen_fake = discriminator(X_p, labels_p)
            
            # train encoder
            enc_loss = latent_loss + discrim_layer_recon_loss
            enc_loss.backward(retain_graph=True)
            optimizer_enc.step()
            
            # train decoder
            dec_loss = -torch.mean(validity_gen_fake) + discrim_layer_recon_loss
            dec_loss.backward(retain_graph=True)
            optimizer_dec.step()
        
        
            if i==0:
                print(
                    "[Epoch %d/%d %ds] [Enc loss: %f] [Dec loss: %f] [Dis loss: %f]"
                    % (epoch+1, opt.n_epochs,  int(time.time()-start), enc_loss.item(), dec_loss.item(), dis_loss.item())
                )
        
                enc_losses.append(enc_loss.item())
                dec_losses.append(dec_loss.item())
                dis_losses.append(dis_loss.item())
                
            batches_done += opt.n_critic
    
        if epoch % 5000 == 0:
            torch.save(encoder.state_dict(), "wvaegan_gp/results/encoder_params_dim{0}_{1}".format(opt.latent_dim, epoch))
            torch.save(decoder.state_dict(), "wvaegan_gp/results/decoder_params_dim{0}_{1}".format(opt.latent_dim, epoch))
            torch.save(discriminator.state_dict(), "wvaegan_gp/results/discriminator_params_dim{0}_{1}".format(opt.latent_dim, epoch))
        if epoch % 5000 == 0:
            sample_image(epoch=epoch)

        

torch.save(encoder.state_dict(), "wvaegan_gp/results/encoder_params_dim{0}_{1}".format(opt.latent_dim, opt.n_epochs+done_epoch))
torch.save(decoder.state_dict(), "wvaegan_gp/results/decoder_params_dim{0}_{1}".format(opt.latent_dim, opt.n_epochs+done_epoch))
torch.save(discriminator.state_dict(), "wvaegan_gp/results/discriminator_params_dim{0}_{1}".format(opt.latent_dim, opt.n_epochs+done_epoch))
sample_image(data_num=100)
save_loss(enc_losses, dec_losses, dis_losses, path="wvaegan_gp/results/loss_dim{0}.png".format(opt.latent_dim))
"""