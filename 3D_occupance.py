# The following code is adapted from the framework of Project wire
# Link to the original framework: https://github.com/vishwa91/wire/tree/main
# Reference: Saragadam Vishwanath et al. WIRE: Wavelet Implicit Neural Representations.
from scipy import io
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import math
import lpips
import tqdm
import warnings
from tensorboardX import SummaryWriter
from utlis import *
from modules import *

if __name__ == '__main__':
    # Load 3D data
    data=io.loadmat('/images/occupance')['hypercube'].astype(np.float32)
    scale=1.0
    mcubes_thres = 0.5          # Threshold for marching cubes
    data=ndimage.zoom(data/data.max(), [scale, scale, scale], order=0)

    # Clip to tightest bound ing box
    hidx, widx, tidx = np.where(data > 0.99)
    data = data[hidx.min():hidx.max(),
                widx.min():widx.max(),
                tidx.min():tidx.max()]
    
    print(data.shape)
    H, W, T = data.shape
    maxpoints=int(5e5) #follow the previous settings
    maxpoints=min(H*W*T, maxpoints)

    dataten=torch.tensor(data).cuda().reshape(H*W*T, 1)

    # creat a model
    mode='sin+fr'
    model=get_INR(mode=mode,in_features=3,
    hidden_features=256,
    hidden_layers=2,
    out_features=1,
    outermost_linear=True,
    high_freq_num=256,
    low_freq_num=256,
    phi_num=8,
    alpha=0.001, #for relu or pe+relu, alpha=0.01
    first_omega_0=30,
    hidden_omega_0=30,
    pe=False
    )
    model=model.cuda()

    # Optimizer and scheduler
    niters=200
    learning_rate=5e-3

    optim=torch.optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler=LambdaLR(optim, lambda x: 0.2**min(x/niters, 1))

    # loss function
    criterion=torch.nn.MSELoss()

    # Create inputs
    coords=get_coords(H, W, T)
    mse_array = np.zeros(niters)
    time_array = np.zeros(niters)
    best_mse = float('inf')
    best_results = None

    tbar=tqdm.tqdm(range(niters))
    im_estim=torch.zeros((H*W*T, 1), device='cuda')

    tic=time.time()
    print('Running %s mode' %mode)
    
    for idx in tbar:
        indices=torch.randperm(H*W*T)

        train_loss=0
        nchunks=0
        for b_idx in range(0, H*W*T, maxpoints):
            b_indices=indices[b_idx:min(H*W*T,b_idx+maxpoints)]
            b_coords=coords[b_indices,...].cuda()
            b_indices=b_indices.cuda()
            pixelvalues=model(b_coords[None, ...]).squeeze()[:, None]

            with torch.no_grad():
                im_estim[b_indices, :] = pixelvalues
            
            loss = criterion(pixelvalues, dataten[b_indices, :])

            optim.zero_grad()
            loss.backward()
            optim.step()

            lossval=loss.item()
            train_loss += lossval
            nchunks += 1
        
        mse_array[idx] = get_IoU(im_estim,dataten,mcubes_thres)
        time_array[idx]=time.time()
        scheduler.step()

        im_estim_vol=im_estim.reshape(H, W, T)

        if lossval < best_mse:
            best_mse = lossval
            best_results = copy.deepcopy(im_estim)

        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
    
    total_time=time.time()-tic
    best_results=best_results.reshape(H,W,T).detach().cpu().numpy()
    io.savemat('results/%s.mat'%(mode))
    print('IoU: ', get_IoU(best_results, data, mcubes_thres))





            





