import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

# Fast Fourier Transform
target_frame=pd.read_csv('results/1d_fitting/target_value.csv')
target_value=target_frame['target']
freq_target=np.abs(np.fft.fft(target_value))
# select different types of outputs
output=pd.read_csv('results/1d_fitting/relu+fr_output.csv')
delta_frame=pd.DataFrame([])
for i in range(10000):
    freq_pred=np.abs(np.fft.fft(output[str(i)]))
    delta_freq=abs(freq_pred-freq_target)/abs(freq_target)
    delta_frame['column'+str(i)]=delta_freq

# We select the principal components, i.e. 1.5Hz,2.5Hz,3.5Hz,4.5Hz.
heat_map_fr=pd.DataFrame([])
heat_map_fr['1.5hz']=delta_frame.iloc[3,:]
heat_map_fr['2.5hz']=delta_frame.iloc[5,:]
heat_map_fr['3.5hz']=delta_frame.iloc[7,:]
heat_map_fr['4.5hz']=delta_frame.iloc[9,:]
heat_map_fr.set_index(np.arange(1,10001),inplace=True)

# Customize the color
custom_colors = [(194, 206, 220), (176, 177, 182), (228, 230, 225)]
custom_colors=['#C2CEDC','#B0B1B6','#E4E6E1']


b=heat_map_fr.values
b=b.T
heat_map_fr=pd.DataFrame(b,columns=heat_map_fr.index)
ax=sns.heatmap(b,cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),annot=False,fmt='.3f')
ax.set_yticklabels(['1.5','2.5','3.5','4.5'])
ax.set_xticks([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
ax.set_xticklabels(np.arange(1,11,1))
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=20
ax.set_title('(b) MLP+ReLU+FR')
ax.set_xlabel('Training Step ($\u00D710^3$)')
ax.set_ylabel('Frequency[Hz]')
plt.savefig('results/1d_fitting/relu_fr.pdf', format='pdf', bbox_inches='tight')
plt.show()
