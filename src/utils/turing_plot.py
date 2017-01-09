import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import numpy as np

def turing_plot(seq, outputs, write_w, read_w):
    """Modified from
    https://github.com/carpedm20/NTM-tensorflow/"""

    length2plus2 = seq.shape[0]
    seq_length = ((length2plus2) - 2) / 2
    seq = seq[1:seq_length+1]
    outputs = outputs.round()[-seq_length:]

    if seq_length >= 80:
        fig = plt.figure(1,figsize=(20,16))
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.4, 0.4, 1.6, 1.6])
    elif seq_length >= 60:
        fig = plt.figure(1,figsize=(20,14))
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.6, 0.6, 1.4, 1.4])
    elif seq_length >= 50:
        fig = plt.figure(1,figsize=(20,14))
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.8, 0.8, 1.2, 1.2])
    elif seq_length >= 20:
        fig = plt.figure(1,figsize=(20,14))
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.9, 0.9, 1.1, 1.1])
    else:
        fig = plt.figure(1,figsize=(20,10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(seq.T, interpolation='nearest')
    ax0.set_ylabel('input')
    
    ax1 = plt.subplot(gs[1])
    ax1.imshow(outputs.T, interpolation='nearest')
    ax1.set_xlabel('time')
    ax1.set_ylabel('output')
    
    ax2 = plt.subplot(gs[2])
    ax2.imshow(write_w[1:-1], cmap='Greys', interpolation='nearest')
    ax2.set_xlabel('write weight')
    ax2.set_ylabel('time')
    
    ax3 = plt.subplot(gs[3])
    ax3.imshow(read_w[1:-1], cmap='Greys', interpolation='nearest')
    ax3.set_xlabel('read weight')
    ax3.set_ylabel('time')

    plt.savefig('input.png')