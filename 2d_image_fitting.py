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
import warnings
from utlis import *
from modules import *
from tqdm.autonotebook import tqdm

if __name__ == '__main__':
    # load image
    koda='05'
    print('Koda'+koda)
    kodim=skimage.io.imread('images/kodim_photo/kodim'+koda+'.png')
    figure= ImageFitting(kodim.shape[0],kodim.shape[1],kodim)
    # dataset
    dataloader=DataLoader(figure,batch_size=1,pin_memory=True,num_workers=0)
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    
    # creat a model
    mode='relu+fr'
    model=get_INR(mode=mode,in_features=2,
        hidden_features=256,
        hidden_layers=3,
        out_features=3,
        outermost_linear=True,
        high_freq_num=128,
        low_freq_num=128,
        phi_num=32,
        alpha=0.05, # for relu, alpha:0.05; for sin, alpha:0.01
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        pe=False   
        )
    
    #optimizer and lr_scheduler
    optim=torch.optim.Adam(lr=1e-4,params=model.parameters())
    scheduler=StepLR(optim,step_size=3000,gamma=0.1)
    total_steps=10000
    model=model.cuda()
    with tqdm(total=total_steps) as pbar:
        max_psnr=0
        for i in range(total_steps):
            model_output=model(model_input)
            loss=((model_output-ground_truth)**2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss=loss.item()
            psnr=10*np.log10(1.0**2/loss)
            if i <=3000:
                scheduler.step()
            max_psnr = max(max_psnr,psnr)
            pbar.set_description(f'Loss: {loss:.4f} | PSNR: {psnr:.2f} | lr: {scheduler.get_last_lr()[0]} ')
            pbar.update(1)
    last_photo=model_output.cpu().view(kodim.shape[0],kodim.shape[1],3).detach().numpy()
    last_photo=np.clip(last_photo,0,1)
    plt.imshow(last_photo)
    plt.savefig('results/2d_image_fitting/results_of_'+str(mode)+'_koda'+koda+'.png')
    print('last psnr:', psnr)
    print('best psnr:', max_psnr)






    

    




