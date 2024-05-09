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
from tensorboardX import SummaryWriter
from utlis import *
from modules import *
from tqdm.autonotebook import tqdm
import time
import cv2
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    # domain interval
    min_val = -1
    max_val = 1
    num_samples=300
    dataset=CustomDataset(min_val,max_val,num_samples)
    dataloader=DataLoader(dataset,batch_size=int(num_samples),shuffle=False)

    # compute the emprical NTK matrix every 1000 epoches to reflect
    # the training dynamic
    ntk_till_summary=1000

    x_train= torch.linspace(min_val, max_val, 300).unsqueeze(-1).to('cuda')
    
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    mode='relu+fr'
    model=get_INR(mode=mode,in_features=1,
        hidden_features=128,
        hidden_layers=3,
        out_features=1,
        outermost_linear=True,
        high_freq_num=64,
        low_freq_num=64,
        phi_num=16,
        alpha=0.05, # for relu, alpha:0.05; for sin, alpha:0.01
        first_omega_0=5.0, 
        hidden_omega_0=5.0, # for 1d function, we set first_omega_0 and hidden_omega_0 as 5.0 for fast convergence.
        pe=False            # As the omega_0 determines the spectrum represented by Siren.
        )
    model=model.cuda()
    
    #optimizer and lr_scheduler
    optim=torch.optim.Adam(lr=1e-6,params=model.parameters())
    total_steps=10000

    loss_list=[]
    eig_value_frame=pd.DataFrame([])

    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)

    #the realization of emprical NTK matrix
    def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = jac1.values()
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = jac2.values()
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        if compute == 'full':
            einsum_expr = 'Naf,Mbf->NMab'
        elif compute == 'trace':
            einsum_expr = 'Naf,Maf->NM'
        elif compute == 'diagonal':
            einsum_expr = 'Naf,Maf->NMa'
        else:
            assert False

        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result
    print(mode)
    loss_values=[]
    output_frame=pd.DataFrame([])
    target_frame=pd.DataFrame([])
    eig_value_frame=pd.DataFrame([])
    start_time = time.time()

    with tqdm(total=total_steps) as pbar:
        params = {k: v.detach() for k, v in model.named_parameters()}
        for i in range(total_steps):
            model_output=model(model_input)
            loss=((model_output-ground_truth)**2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            # If you want to explore the detailed training time, please
            # comment the following NTK code block.
            if not i % ntk_till_summary:
                ntk_matrix=empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_train, 'full')
                ntk_matrix=ntk_matrix[:,:,0,0]
                eig_value=torch.linalg.eigvals(ntk_matrix)
                eig_value=eig_value.detach().cpu().numpy()
                # In theory, as the jacobian matrix of the emprical NTK matrix is a Gram matrix,
                # the eigenvalues of this matrix are always positive. However, numerical precision issue in
                # computation can lead to near-zaro eigenvalues resulting in negative values or imaginary parts 
                # after calculation. Hence, the magnitude is adopted and it will not affect the original accurate 
                # eigenvalue.
                loss_values.append(loss.item())
                eig_value_frame[str(i)]=np.abs(eig_value)
            output_frame[str(i)]=model_output.detach().cpu().numpy()[:,0]
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
    end_time = time.time()
    print('Training time/epoch: ', (end_time - start_time)/total_steps)
    eig_value_frame.to_csv('results/1d_fitting/'+str(mode)+'_eig_value_frame.csv')
    output_frame.to_csv('results/1d_fitting/'+str(mode)+'_output.csv')
    target_frame['target']=ground_truth.detach().cpu().numpy()[:,0]
    target_frame.to_csv('results/1d_fitting/target_value.csv')
    


#The three largest eigenvalues are selected, and the remaining eigenvalues are summed.
for i in range(total_steps//ntk_till_summary):
    first_eigen=eig_value_frame.iloc[0,i]
    second_eigen=eig_value_frame.iloc[1,i]
    third_eigen=eig_value_frame.iloc[2,i]
    fourth_eigen=eig_value_frame.iloc[3,i]
    remaining_eigen=np.sum(eig_value_frame.iloc[4:,i],axis=0)
    sum_eign=first_eigen+second_eigen+third_eigen+fourth_eigen+remaining_eigen
    categories=['First','Second','Third','Fourth','Remain']
    values=[first_eigen/sum_eign,second_eigen/sum_eign,third_eigen/sum_eign,fourth_eigen/sum_eign,remaining_eigen/sum_eign]
    plt.clf()
    plt.ylim(0,1)
    plt.bar(categories,values)
    plt.title('Percentage of '+mode+' NTK eigenvalues')
    plt.show()
    plt.savefig('results/dynamic_eNTK/Epoch'+str(i)+'_'+mode+'_NTK_eigenvalues.png')





    






            


