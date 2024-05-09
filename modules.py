import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import math
import lpips
import warnings

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class Fourier_reparam_linear(nn.Module):
    def __init__(self,in_features,out_features,high_freq_num,low_freq_num,phi_num,alpha):
        super(Fourier_reparam_linear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) # 取最大周期作为取点区间
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases

    
    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator,np.sqrt(6/m)/dominator)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)
        
    def forward(self,x):
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return output

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class sin_fr_layer(nn.Module):
    def __init__(self, in_features, out_features, high_freq_num,low_freq_num,phi_num,alpha,omega_0=30.0):
        super().__init__()
        super(sin_fr_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.omega_0=omega_0
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) # 取最大周期作为取点区间
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases

    
    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator/self.omega_0,np.sqrt(6/m)/dominator/self.omega_0)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)
        
    def forward(self,x):
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x,weight.transpose(0,1))
        output=output+self.bias.T
        return torch.sin(self.omega_0*output)


class INR(nn.Module):
    def __init__(self,mode,in_features,hidden_features,hidden_layers,out_features,outermost_linear,high_freq_num,low_freq_num,
    phi_num,alpha,first_omega_0,hidden_omega_0,pe):
        super().__init__()
        self.net=[]
        self.pe=pe
        if pe==True:
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,sidelength=256,fn_samples=None,use_nyquist=True)
            in_features=self.positional_encoding.out_dim
        self.net=[]
        if mode=='relu':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())
        
        if mode=='relu+fr':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(Fourier_reparam_linear(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha))
                self.net.append(nn.ReLU())
        
        if mode=='relu+pe':
            self.net.append(nn.Linear(in_features,hidden_features))
            self.net.append(nn.ReLU())
            for i in range(hidden_layers):
                self.net.append(nn.Linear(hidden_features,hidden_features))
                self.net.append(nn.ReLU())

        if mode=='sin':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(SineLayer(hidden_features, hidden_features,is_first=False, omega_0=hidden_omega_0))
        
        if mode=='sin+fr':
            self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))
            for i in range(hidden_layers):
                self.net.append(sin_fr_layer(hidden_features,hidden_features,high_freq_num,low_freq_num,phi_num,alpha,hidden_omega_0))
        #末端初始化这边还是需要修改
        if outermost_linear==True:
            final_linear = nn.Linear(hidden_features, out_features)
            if mode =='sin+fr' or mode=='sin':
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features)/hidden_omega_0,np.sqrt(6 / hidden_features)/hidden_omega_0)
            else:
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6/ hidden_features),np.sqrt(6 / hidden_features)) 
            self.net.append(final_linear)
        else:
            if mode=='relu' or mode=='relu+fr':
                final_linear=nn.Linear(hidden_features,out_features)
                self.net.append(final_linear)
                self.net.append(nn.ReLU())
            if mode=='sin' or mode=='sin+fr':
                self.net.append(SineLayer(hidden_features, out_features,is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        if self.pe==True:
            x=self.positional_encoding(x)
        output=self.net(x)
        return output

def get_INR(mode,in_features, hidden_features, hidden_layers,
            out_features, outermost_linear, high_freq_num,low_freq_num,phi_num,alpha,first_omega_0,
            hidden_omega_0, pe):
    '''
        Function to get a class instance for a given type of
        implicit neural representation
        
        Inputs:
            mode: non-linear activation functions and Fourier reparameterized training
            in_features: Number of input features. 2 for image, 3 for volume and so on.
            hidden_features: Number of features per hidden layer
            hidden_layers: Number of hidden layers
            out_features; Number of outputs features. 3 for color image, 1 for grayscale or volume and so on
            outermost_linear (True): If True, do not apply nonlin
                just before output
            high_freq_num: The number of the high frequence in the Fourier bases B. 
            low_freq_num: The number of the low frequence in the Fourier bases B.
            phi_num: The number of the phase in the Fourier bases B.
            (high_freq_num, low_freq_num, phi_num): The detailed description can be found in Sec. 3.2 of our paper
            alpha: The role can be found in Appendix A. Detailed proof of Theorem 2 of our paper. Empirically, alpha=0.05 for relu; 
                alpha=0.01 for sin and so on.
            first_omega0 (30): For siren: Omega for first layer
            hidden_omega0 (30): For siren and siren+fr: Omega for hidden layers
            pos_encode (False): If True apply positional encoding
        Output: An INR class instance
    '''
    model=INR(mode, in_features, hidden_features, hidden_layers, out_features, outermost_linear, high_freq_num, low_freq_num, phi_num, alpha, first_omega_0, hidden_omega_0, pe)
    
    return model

