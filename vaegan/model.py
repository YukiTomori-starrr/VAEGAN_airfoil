#---------------------------------------------------
# vae/genのモデル定義
#---------------------------------------------------
 
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

coord_shape = (1, 496)

# input:(64, 496), output:(64, latent_dim)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(coord_shape[1], 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, latent_dim),
            nn.Tanh()
        )
        self.l_mu = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.l_var = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        
    def forward(self, noise):
        latent = self.model(noise)
        latent = latent.view(latent.shape[0], 3)
        mu = self.l_mu(latent)
        logvar = self.l_var(latent)
        return mu, logvar
    
    
# input:(64, latent_dim+1), output:(64, 1, 496)
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *block(latent_dim+1, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, 496),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), -1).to("cuda")
        coords = self.model(gen_input).to("cuda")
        coords = coords.view(coords.shape[0], *coord_shape)
        return coords
        
        
# input:(64, 497), output:(64, 1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(in_feat, out_feat, dropout=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if dropout:
                layers.append(nn.Dropout(0.4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(1 + 496, 512, dropout=False),
            # *block(512, 512),
            *block(512, 256, dropout=False),
            # *block(256, 128),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, coords, labels):
        # Concatenate label embedding and image to produce input
        c_coords = torch.cat((coords.view(coords.size(0), -1), labels), -1)
        c_coords_flat = c_coords.view(c_coords.shape[0], -1)
        validity = self.model(c_coords_flat)
        return validity
