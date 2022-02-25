if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

from wvaegan_gp.model_nonvector import Decoder, Encoder
import matplotlib.pyplot as plt
from calc_cl import get_cl, get_cls
from util import to_cpu, to_cuda, save_coords_by_cl 

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
coord_shape = (1, 496)

class Eval:
  def __init__(self, Dec_PATH, Enc_PATH, coords_npz):
    #state_dict = torch.load(Dec_PATH, map_location=torch.device('cpu'))
    state_Dec_dict = torch.load(Dec_PATH, map_location=torch.device('cuda'))
    self.Dec = Decoder(latent_dim).to("cuda")
    self.Dec.load_state_dict(state_Dec_dict)
    self.Dec.eval()
    state_Enc_dict = torch.load(Enc_PATH, map_location=torch.device('cuda'))
    self.Enc = Encoder(latent_dim).to("cuda")
    self.Enc.load_state_dict(state_Enc_dict)
    self.Enc.eval()
    self.latent_dim = 3 #change
    self.coords = {
      'data': coords_npz[coords_npz.files[0]],
      'mean':coords_npz[coords_npz.files[1]],
      'std':coords_npz[coords_npz.files[2]],
    }

  def rev_standardize(self, coords):
    return coords*self.coords['std']+self.coords['mean']

  def create_coords_by_cl(self, cl_c, data_num=20):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, self.latent_dim))))
    labels = np.array([cl_c]*data_num)
    labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, 1)))
    #gen_coords = self.rev_standardize(to_cpu(self.Dec(z, labels)).detach().numpy()
    gen_coords = self.rev_standardize(self.Dec(z, labels).cpu().detach().numpy())
    return gen_coords

  def create_successive_coords(self):
    """0.01から1.50まで151個のC_L^cと翼形状を生成"""
    cl_r = []
    cl_c = []
    dec_coords = []
    cl_list = []
    for cl in range(151):
      cl /= 100
      cl_c.append(cl)
      labels = Variable(torch.reshape(FloatTensor([cl]), (1, 1)))
      calc_num = 0
      while (True):
        calc_num += 1
        z = Variable(FloatTensor(np.random.normal(0, 1, (1, coord_shape[1]))))
        #dec_coord = self.rev_standardize(to_cpu(self.Dec(z, labels)).detach().numpy())
        mus, log_variances = self.Enc(z, self.latent_dim)
        variances = torch.exp(log_variances * 0.5)
        #修正箇所12/28,ここから
        Z_p = Variable(FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
        #ここまで
        Z = Z_p * variances + mus
        
        dec_coord = self.rev_standardize(self.Dec(Z, labels).cpu().detach().numpy())
        clr = get_cl(dec_coord)
        # cl = 0.1
        if not np.isnan(clr):
          print(cl)
          cl_r.append(clr)
          dec_coords.append(dec_coord)
          cl_list.append(np.linalg.norm(cl - clr)**2)
          break
        if calc_num == 5:
          print('not calculated {0}'.format(cl))
          cl_r.append(-1)
          dec_coords.append(dec_coord)
          break

    np.savez("wvaegan_gp/results/successive_label_dim{0}".format(self.latent_dim), cl_c, cl_r, dec_coords)
    np.savez("wvaegan_gp/results/calc_mse_dim{0}".format(self.latent_dim),cl_list)
    return cl_list

  def save_coords(self, dec_coords, labels, path):
    data_size = dec_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = dec_coords[i]
        label = labels[i]
        x,y = coord.reshape(2, -1)
        ax[i%4, i//4].plot(x,y)
        cl = round(label.item(), 4)
        title = 'CL={0}'.format(str(cl))
        ax[i%4, i//4].set_title(title)

    fig.savefig(path)

  def successive(self):
    coords_npz = np.load("wvaegan_gp/results/successive_label_dim{0}.npz".format(self.latent_dim))
    cl_c = coords_npz[coords_npz.files[0]]
    cl_r = coords_npz[coords_npz.files[1]]
    success_clc = []
    success_clr = []
    fail_clc = []
    fail_clr = []
    for c, r in zip(cl_c, cl_r):
      if r == -1:
        fail_clc.append(c)
        fail_clr.append(0)
        continue
      success_clc.append(c)
      success_clr.append(r)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1.5])
    x = np.linspace(0, 1.5, 10)
    ax.plot(x, x, color = "black")
    ax.scatter(success_clc, success_clr)
    ax.scatter(fail_clc, fail_clr, color='red')
    ax.set_xlabel("Specified label")
    ax.set_ylabel("Recalculated label")
    # plt.show()
    fig.savefig("wvaegan_gp/results/successive_label_dim{0}.png".format(self.latent_dim))

  def sample_data(self, data_num=100):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, 3))))
    labels = 1.558*np.random.random_sample(size=(data_num, 1))
    labels = Variable(FloatTensor(labels))
    #dec_coords = to_cpu(self.Dec(z, labels)).detach().numpy()
    #labels = to_cpu(labels).detach().numpy()
    dec_coords = to_cuda(self.Dec(z, labels)).detach().numpy()
    labels = to_cuda(labels).detach().numpy()
    np.savez("wvaegan_gp/results/final_dim{0}".format(self.latent_dim), labels,self.rev_standardize(dec_coords))

  def euclid_dist(self, coords):
    """mu, バリエーションがどれぐらいあるか"""
    mean = np.mean(coords, axis=0)
    mu_d = np.linalg.norm(coords - mean)/len(coords)
    return mu_d
  
  def mu_dist(self, coords):
    mean = np.mean(coords, axis=0)
    mu_d = (np.linalg.norm(coords/len(coords)-mean))**2*len(coords)
    return mu_d

  def _dist_from_dataset(self, coord):
    """データセットからの距離の最小値"""
    min_dist = 100
    idx = -1
    for i, data in enumerate(self.rev_standardize(self.coords['data'])):
      dist = np.linalg.norm(coord - data)
      if dist < min_dist:
        min_dist = dist
        idx = i
    
    return min_dist, idx
    
  def calc_dist_from_dataset(self, coords, clr):
    data_idx = -1
    decode_idx = -1
    max_dist = 0
    for i, c in enumerate(coords):
      cl = clr[i]
      if not np.isnan(cl):
        dist, didx = self._dist_from_dataset(c)
        if dist > max_dist:
          max_dist = dist
          data_idx = didx
          decode_idx = i
    return max_dist, data_idx, decode_idx
  
  def calc_mse(self, coords, cl_c):
    print((cl_c - clr)**2)
    return np.linalg.norm(cl_c - clr)**2/len(coords)
  
  def notconverge_rate(self):
    for i in range(151):
      i /= 100
      cl_c = round(i, 2)
      coords = evl.create_coords_by_cl(cl_c)
      coords = coords.reshape(coords.shape[0], -1)
      clr = get_cls(coords)
      not_converge = np.count_nonzero(np.isnan(clr))
      notconverge_rate = not_converge/len(clr)
      failure = np.count_nonzero(abs(clr-cl_c)>0.2)
      failure_rate = failure/len(clr)
      print("cl,{0},{1},{2}, {3}".format(cl_c, notconverge_rate, failure_rate, 1-notconverge_rate-failure_rate))

if __name__ == "__main__":
  # -------------------------
  #  fix 
  # -------------------------
  latent_dim = 3
  coords_npz = np.load("dataset/standardized_coords.npz")
  perfs = np.load("dataset/perfs.npy")
  Dec_PATH = "wvaegan_gp/results/decoder_params_dim{0}_50000".format(latent_dim)
  Enc_PATH = "wvaegan_gp/results/encoder_params_dim{0}_50000".format(latent_dim)
  print(Dec_PATH)
  #Dec_PATH = "wvaegan_gp/results/decoder_params_10000".format(latent_dim)
  evl = Eval(Dec_PATH, Enc_PATH, coords_npz)
  # -------------------------
  #  free
  # -------------------------
  """
  cl_c = 1.4
  coords = evl.create_coords_by_cl(cl_c)
  coords = coords.reshape(coords.shape[0], -1)
  mu = evl.euclid_dist(coords)
  print(mu, len(coords))
  
  clr = get_cls(coords)
  #evl.save_coords(coords, clr, "wvaegan_gp/results/coords_{0}.png".format(cl_c))
  max_dist, d_idx, g_idx = evl.calc_dist_from_dataset(coords, clr)
  print(max_dist)
  d_coord = evl.rev_standardize(evl.coords['data'][d_idx])
  d_cl = perfs[d_idx]
  g_coord = coords[g_idx]
  g_cl = clr[g_idx]
  print(cl_c, d_cl, g_cl)
  #cls = np.array([cl_c, d_cl, g_cl])
  """
  #mse_list = np.load("wvaegan_gp/results/calc_mse_dim{0}.npz".format(latent_dim))
  #print(mse_list.shape)
  #np.savez("vaegan/dist_{0}".format(cl_c), d_coord, g_coord, cls, max_dist)
  mse_list = evl.create_successive_coords()
  #evl.save_coords("vaegan/results/successive_label.npz", perfs, "vaegan/results/coords")
  evl.successive()
  print(sum(mse_list)/len(mse_list))
  
  """
  for i in range(151):
    i /= 100
    cl_c = round(i, 2)
    coords = evl.create_coords_by_cl(cl_c)
    coords = coords.reshape(coords.shape[0], -1)
    clr = get_cls(coords)
    failure = np.count_nonzero(abs(clr-cl_c)>0.2)
    failure_rate = failure/len(clr)
    not_converge = np.count_nonzero(np.isnan(clr))
    notconverge_rate = not_converge/len(clr)
    success_rate = 1-failure_rate-notconverge_rate
    
    print("cl,{0},failure,{1},notconverge,{2},success,{3}".format(cl_c, failure_rate, notconverge_rate, success_rate))
  """
  """
  for i in range(151):
    i /= 100
    cl_c = round(i, 2)
    coords = evl.create_coords_by_cl(cl_c)
    coords = coords.reshape(coords.shape[0], -1)
    clr = get_cls(coords)
    mu = evl.euclid_dist(coords)
    #failure = np.count_nonzero(abs(clr-cl_c)>0.2)
    #failure_rate = failure/len(clr)
    #not_converge = np.count_nonzero(np.isnan(clr))
    #notconverge_rate = not_converge/len(clr)
    
    print("cl,{0},{1}".format(cl_c, mu))
  """