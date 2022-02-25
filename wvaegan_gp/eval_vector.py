# --------------------------------------------------------------------------------------
# Encoder includes labels imfomation ver
# --------------------------------------------------------------------------------------
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

from wvaegan_gp.model_vector import Decoder, Encoder
import matplotlib.pyplot as plt
from calc_cl import get_cl, get_cls
from util import to_cpu, to_cuda, save_coords_by_cl 
import math

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
    self.latent_dim = 4 #change
    self.coords = {
      'data': coords_npz[coords_npz.files[0]],
      'mean':coords_npz[coords_npz.files[1]],
      'std':coords_npz[coords_npz.files[2]],
    }

  def rev_standardize(self, coords):
    return coords*self.coords['std']+self.coords['mean']

  def create_coords_by_cl(self, cl_c, data_num=20):
    labels = np.array([cl_c]*data_num)
    labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, 1)))
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, coord_shape[1]))))
    #データセットをencode
    mus, log_variances = self.Enc(z, labels, self.latent_dim)
    variances = torch.exp(log_variances * 0.5)
    Z_p = Variable(FloatTensor(np.random.normal(0, 1, (data_num, self.latent_dim))))
    Z = Z_p * variances + mus
    gen_coords = self.rev_standardize(self.Dec(Z, labels).cpu().detach().numpy())
    return gen_coords

  def create_successive_coords(self):
    """0.01から1.50まで151個のC_L^cと翼形状を生成"""
    cl_r = []
    cl_c = []
    dec_coords = []
    cl_list = []
    for cl in range(1501):
      cl /= 1000
      cl_c.append(cl)
      labels = Variable(torch.reshape(FloatTensor([cl]), (1, 1)))
      calc_num = 0
      while (True):
        calc_num += 1
        z = Variable(FloatTensor(np.random.normal(0, 1, (1, coord_shape[1]))))
        #データセットをencode
        mus, log_variances = self.Enc(z, labels, self.latent_dim)
        variances = torch.exp(log_variances * 0.5)
        Z_p = Variable(FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
        Z = Z_p * variances + mus
        dec_coord = self.rev_standardize(self.Dec(Z, labels).cpu().detach().numpy())
        clr = get_cl(dec_coord)
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

    np.savez("wvaegan_gp/results/successive_label_dim{0}_vector".format(self.latent_dim), cl_c, cl_r, dec_coords)
    #np.savez("wvaegan_gp/results/calc_mse_merged_dim{0}_vector".format(self.latent_dim),cl_list)
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
    coords_npz = np.load("wvaegan_gp/results/successive_label_dim{0}_vector.npz".format(self.latent_dim))
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
    fig.savefig("wvaegan_gp/results/successive_label_merged_dim{0}_vector.png".format(self.latent_dim))

  def sample_data(self, data_num=100):
    labels = 1.558*np.random.random_sample(size=(data_num, 1))
    labels = Variable(FloatTensor(labels))
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, coord_shape[1]))))
    #データセットをencode
    mus, log_variances = self.Enc(z, labels, self.latent_dim)
    variances = torch.exp(log_variances * 0.5)
    #修正箇所12/28,ここから
    Z_p = Variable(FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
    Z = Z_p * variances + mus
    
    dec_coords = to_cuda(self.Dec(Z, labels).detach().numpy())
    np.savez("wvaegan_gp/results/final_dim{0}".format(self.latent_dim), labels,self.rev_standardize(dec_coords))

  # --------------------------------------------
  # muを計算する関数
  # --------------------------------------------
  def euclid_dist(self, coords):
    coords = coords.reshape(coords.shape[0], -1)
    mean = np.mean(coords, axis=0)
    diff = (coords-mean)**2
    mu_d = np.sqrt(np.sum(diff, axis=1))
    mu = np.mean(mu_d)
    return mu
  
  def mu_dist(self, coords):
    """バリエーションがどれぐらいあるか"""
    mean = np.mean(coords, axis=0)
    mu_d = np.linalg.norm(coords - mean)/len(coords)
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

  # -----------------------------------------
  # 生成翼型を重ねて表示する関数
  # -----------------------------------------
  def plot_multicoords(self, clc):
    coords = evl.create_coords_by_cl(clc)
    fig = plt.figure()
    plt.title("CL:{0}".format(clc))
    for i in range(len(coords)):
        coord = coords[i]
        x,y = coord.reshape(2, -1)
        plt.plot(x,y)
    fig.savefig("wvaegan_gp/coords/vaegan_multifig_{0}.png".format((clc)))
  
  # -----------------------------------------
  # MSE誤差
  # -----------------------------------------
  def calc_mse(self, cl_c, clr):
    new_clr = np.array([x for x in clr if math.isnan(x) == False])
    data_num = len(new_clr)
    new_clc = np.array([cl_c]*data_num)

    diff = np.abs(new_clr - new_clc)**2
    mse = np.mean(diff)
    return mse
      
  # ---------------------------------------
  # smoothnessを測る関数（単体）
  # ---------------------------------------
  def smoothness(self, coord):
    x, y = coord.reshape(2, -1)
    x_diff = np.diff(x, n=1)
    x_diff = np.append(x_diff, x[0]-x[len(x)-1])
    y_diff = np.diff(y, n=1)
    y_diff = np.append(y_diff, y[0]-y[len(y)-1])
    v = np.array(list(zip(x_diff, y_diff)))
    # calculate phi
    phi_list = []
    for i in range(len(v)):
      if i != len(v)-1:
        phi = math.acos(np.dot(v[i], v[i+1]) / (np.linalg.norm(v[i])*np.linalg.norm(v[i+1])))
        phi_list.append(phi)
      if i == len(v)-1:
        phi = math.acos(np.dot(v[i], v[0]) / (np.linalg.norm(v[i])*np.linalg.norm(v[0])))
        phi_list.append(phi)
    return(sum(phi_list))
  
  # ----------------------------------------
  # smoothnessを測る関数（全体）
  # ----------------------------------------
  def calc_smoothness(self, coords):
    coord_list = []
    for coord in coords:
      phi = self.smoothness(coord)
      coord_list.append(phi)
    return sum(coord_list)/len(coord_list)/math.pi
  
  # ----------------------------------------
  # converge, failure, smoothness, mse, muを算出する関数
  # ----------------------------------------
  def calc_values(self):
    for i in range(151):
      i /= 100
      cl_c = round(i, 2)
      coords = evl.create_coords_by_cl(cl_c)
      coords = coords.reshape(coords.shape[0], -1)
      
      clr = get_cls(coords)
      
      converge = 1-(np.count_nonzero(np.isnan(clr)))/len(clr)
      failure = (np.count_nonzero(abs(clr-cl_c)>0.2))/len(clr)
      success = converge-failure
      mu = evl.euclid_dist(coords)
      smoothness = evl.calc_smoothness(coords)
      mse =  evl.calc_mse(cl_c, clr)
    
      print("cl,{0},converge,{1},failure,{2}, success,{3},smoothness,{4},mse,{5}, mu,{6}".format(cl_c, converge, failure, success, smoothness, mse, mu))
      
      

if __name__ == "__main__":
  # -------------------------
  #  fix 
  # -------------------------
  latent_dim = 4
  coords_npz = np.load("dataset/standardized_upsampling_coords.npz")
  perfs = np.load("dataset/upsampling_perfs.npy")
  coords = coords_npz[coords_npz.files[0]]
  coord_mean = coords_npz[coords_npz.files[1]]
  coord_std = coords_npz[coords_npz.files[2]]
  
  Dec_PATH = "wvaegan_gp/results/decoder_params_dim{0}_vector_50000".format(latent_dim)
  Enc_PATH = "wvaegan_gp/results/encoder_params_dim{0}_vector_50000".format(latent_dim)
  evl = Eval(Dec_PATH, Enc_PATH, coords_npz)
  
  
  # -------------------------
  #  free
  # -------------------------

  """
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
  #"""
  """
  #np.savez("vaegan/dist_{0}".format(cl_c), d_coord, g_coord, cls, max_dist)
  #evl.save_coords("vaegan/results/successive_label.npz", perfs, "vaegan/results/coords")
  evl.successive()
  """
  #evl.calc_values()  
  #evl.plot_multicoords(0.68) 