#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os


from convolution import conv2d

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


# In[2]:


matrix = np.array([[1, 4, 4, 2, 1, 0, 0, 1, 0, 0, 3, 3, 3, 4], 
                   [0, 2, 0, 2, 0, 3, 4, 4, 2, 1, 1, 3, 0, 4],
                   [1, 1, 0, 0, 3, 4, 2, 4, 4, 2, 3, 0, 0, 4],
                   [4, 0, 1, 2, 0, 2, 0, 3, 3, 3, 0, 4, 1, 0],
                   [3, 0, 0, 3, 3, 3, 2, 0, 2, 1, 1, 0, 4, 2],
                   [2, 4, 3, 1, 1, 0, 2, 1, 3, 4, 4, 0, 2, 3],
                   [2, 4, 3, 3, 2, 1, 4, 0, 3, 4, 1, 2, 0, 0],
                   [2, 1, 0, 1, 1, 2, 2, 3, 0, 0, 1, 2, 4, 2],
                   [3, 3, 1, 1, 1, 1, 4, 4, 2, 3, 2, 2, 2, 3]])
matrix.shape


# In[3]:


kernel = np.array([[0, 1, 3, 3, 2], 
                   [0, 1, 3, 1, 3],
                   [1, 1, 2, 0, 2],
                   [2, 2, 3, 2, 0],
                   [1, 3, 1, 2, 0]])
kernel.shape


# In[4]:


fig = plt.figure(figsize=(14, 21))
gs = gridspec.GridSpec(9, 21, figure=fig)

ax1 = fig.add_subplot(gs[:, :14])
ax1.set_title('Matrix')
ax1.tick_params(left=False, bottom=False)
ax2 = fig.add_subplot(gs[:, 16:])
ax2.set_title('Kernel')
ax2.tick_params(left=False, bottom=False)

sns.heatmap(matrix, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4, ax=ax1)
sns.heatmap(kernel, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4, ax=ax2)

plt.savefig('files/plot_random_matrix_and_kernel.jpg', bbox_inches='tight')


# In[7]:


feature_map = conv2d(matrix, kernel, stride=(2, 1), dilation=(1, 2))

fig, ax = plt.subplots(figsize=(4,4))#[max(feature_map.shape)]*2))

ax = sns.heatmap(feature_map, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4, fmt='.3g')
ax.tick_params(left=False, bottom=False)
ax.set_title('Obtained Feature map')
plt.ylim(plt.ylim()[0]+0.5, plt.ylim()[1]-0.5)
plt.savefig('files/plot_random_feature_map.jpg', bbox_inches='tight')


# In[11]:


receptive_field_0 = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0],
                     [0, 1], [2, 1], [4, 1], [6, 1], [8, 1],
                     [0, 2], [2, 2], [4, 2], [6, 2], [8, 2],
                     [0, 3], [2, 3], [4, 3], [6, 3], [8, 3],
                     [0, 4], [2, 4], [4, 4], [6, 4], [8, 4]]


# In[13]:



fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(14, 28, figure=fig)
for j_i, j in enumerate([0, 2, 4]):
    for i_i, i in enumerate([0, 1, 2, 3, 4, 5]):
        fig.clear()
    # Highlighted matrix
        ax1 = fig.add_subplot(gs[:9, :14])
        ax1.set_title('Matrix')
        ax1.tick_params(left=False, bottom=False)

        ax1 = sns.heatmap(matrix, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4)
        for x, y in receptive_field_0:
            ax1.add_patch(Rectangle([x+i, y+j], 1, 1, fill=False, edgecolor='yellow', lw=2))

        # Submatrix
        submatrix = np.array([matrix[y+j, x+i] for x, y in receptive_field_0]).reshape(5, 5)

        ax2 = fig.add_subplot(gs[:9, 16:21])
        ax2.set_title('Highlighted submatrix')
        ax2.tick_params(left=False, bottom=False)

        ax2 = sns.heatmap(submatrix, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4)
        ax2.add_patch(Rectangle([0, 0], 5, 5, fill=False, edgecolor='yellow', lw=2))

        # Kernel
        ax3 = fig.add_subplot(gs[:9, 23:])
        ax3.set_title('Kernel')
        ax3.tick_params(left=False, bottom=False)
        ax3 = sns.heatmap(kernel, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4)

        # Obtained Feature Map
        feature_map = conv2d(matrix, kernel, stride=(2, 1), dilation=(1, 2))

        ax4 = fig.add_subplot(gs[6:, 11:17])
        ax4.set_title('Obtained Feature Map')
        ax4.tick_params(left=False, bottom=False)

        ax4 = sns.heatmap(feature_map, cbar=False, annot=True, square=True, cmap='Blues', vmin=-4)
        ax4.add_patch(Rectangle([i_i, j_i], 1, 1, fill=False, edgecolor='yellow', lw=2))

        plt.savefig(f'files/plot_convolution_process_{j}_{i}.jpg', bbox_inches='tight')

os.system(f'ffmpeg -hide_banner -loglevel warning -pattern_type glob -r 1 -i "files/plot_convolution_process_*.jpg" files/convolution_process.gif')
os.system('rm -rf files/plot_convolution_process_*')