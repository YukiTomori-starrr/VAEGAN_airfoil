if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from numpy.lib.type_check import real
from torch.autograd import Variable
import torch
import torch.nn as nn

from wvaegan_gp.model_nonvector import Decoder, Encoder
import matplotlib.pyplot as plt
from calc_cl import get_cl, get_cls
from util import to_cpu, to_cuda, save_coords_by_cl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import seaborn as sns

sns.set_style("darkgrid")

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Eval:
  def __init__(self, Enc_PATH, Dec_PATH, coords_npz):
    dec_state_dict = torch.load(Dec_PATH, map_location=torch.device('cuda'))
    enc_state_dict = torch.load(Enc_PATH, map_location=torch.device('cuda'))
    self.Dec = Decoder(3).to("cuda")
    self.Dec.load_state_dict(dec_state_dict)
    self.Dec.eval()
    self.Enc = Encoder(3).to("cuda")
    self.Enc.load_state_dict(enc_state_dict, strict=False)
    self.Enc.eval()
    self.latent_dim = 3
    self.coords = {
      'data': coords_npz[coords_npz.files[0]],
      'mean':coords_npz[coords_npz.files[1]],
      'std':coords_npz[coords_npz.files[2]],
    }
    

  def rev_standardize(self, coords):
    return coords*self.coords['std']+self.coords['mean']
    
  def visualize_latentspace(self, dim, latent_vector):
    latent_vector = latent_vector.cpu().detach().numpy()
    if dim==3:
      # ------------------------------------------------------------
      # 3 dimension
      # ------------------------------------------------------------
      # visualize
      df = pd.DataFrame(perfs.detach().numpy(), columns=["labels"])
      X = latent_vector[:, 0]
      Y = latent_vector[:, 1]
      Z = latent_vector[:, 2]
      #カラーマップ
      cm = plt.cm.get_cmap('RdYlBu')
      fig = plt.figure()
      ax = Axes3D(fig)
      mappable = ax.scatter(X, Y, Z, c=df["labels"], cmap=cm)
      fig.colorbar(mappable, ax=ax)
      plt.show()
    if dim==2:
      # -------------------------------------------------------------
      # 2 dimension
      # -------------------------------------------------------------
      # Reduce dimension by t-sne
      latent_vecs = TSNE(n_components=2).fit_transform(latent_vector)
      labels = perfs.cpu().detach().numpy()
      # visualize
      X = latent_vecs[:, 0]
      Y = latent_vecs[:, 1]
      #カラーマップ
      cm = plt.cm.get_cmap('RdYlBu')
      fig = plt.figure()
      ax = fig.add_subplot(111)
      mappable = ax.scatter(X, Y, c=labels, cmap=cm)
      fig.colorbar(mappable, ax=ax)
      plt.show()

if __name__ == "__main__":
  #coords_npz = np.load("dataset/standardized_coords.npz")
  #perfs = np.load("dataset/perfs.npy")
  
  # Configure data loader
  perfs_npz = np.load("dataset/standardized_upsampling_perfs.npz")
  coords_npz = np.load("dataset/standardized_upsampling_coords.npz")
  coords = coords_npz[coords_npz.files[0]]
  coord_mean = coords_npz[coords_npz.files[1]]
  coord_std = coords_npz[coords_npz.files[2]]
  perfs = perfs_npz[perfs_npz.files[0]]
  perf_mean = perfs_npz[perfs_npz.files[1]]
  perf_std = perfs_npz[perfs_npz.files[2]]
  
  Enc_PATH = "wvaegan_gp/results/encoder_params_dim3_50000"
  Dec_PATH = "wvaegan_gp/results/decoder_params_dim3_50000"
  evl = Eval(Enc_PATH, Dec_PATH, coords_npz)
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
  parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
  parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate") # 1e-4
  parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient") # 0.0
  parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient") # 0.9
  parser.add_argument("--latent_dim", type=int, default=3, help="dimensionality of the latent space")
  parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
  parser.add_argument("--coord_size", type=int, default=496, help="size of each image dimension")
  parser.add_argument("--channels", type=int, default=1, help="number of image channels")
  parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
  # parser.add_argument("--sample_interval", type=int, default=10000, help="interval betwen image samples")
  opt = parser.parse_args()

  coord_shape = (opt.channels, opt.coord_size)
  
  # Configure input
  coords = torch.tensor(coords)
  perfs = torch.tensor(perfs)
  real_imgs = to_cuda(Variable(coords.type(FloatTensor)))
  labels = to_cuda(Variable(torch.reshape(perfs.float(), (coords.shape[0], opt.n_classes)))).cpu().detach().numpy()
  
  log_variances, mu = evl.Enc(real_imgs, labels, latent_dim=opt.latent_dim)
  variances = torch.exp(log_variances * 0.5)
  
  evl.visualize_latentspace(dim=3, latent_vector=variances)
