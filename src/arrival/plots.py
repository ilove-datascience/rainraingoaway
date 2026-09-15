"""Notebook figures for ordinal arrival probabilities and calibration."""
import numpy as np
import matplotlib.pyplot as plt
from arrival.data import CLASSES


def plot_history(history):
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot([r['step'] for r in history],[r['train_loss'] for r in history],label='Training NLL (balanced sampling)')
    ax.plot([r['step'] for r in history],[r['validation_nll'] for r in history],label='Validation NLL (natural sampling)')
    ax.set(xlabel='Optimizer updates',ylabel='NLL');ax.legend();plt.show()
    if history and 'validation_ap' in history[0]:
        fig,ax=plt.subplots(figsize=(8,3))
        for horizon in ('15','30','60'):
            ax.plot([r['step'] for r in history],[r['validation_ap'][horizon] for r in history],label=f'Within {horizon} min')
        ax.set(xlabel='Optimizer updates',ylabel='Validation average precision');ax.legend();plt.show()


def plot_evaluation(probabilities,labels,result):
    p=np.asarray(probabilities);y=np.asarray(labels)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    confusion=np.asarray(result['confusion']);denominator=confusion.sum(1,keepdims=True)
    fractions=np.divide(confusion,denominator,out=np.zeros_like(confusion,dtype=float),where=denominator>0)
    im=axes[0].imshow(fractions,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=axes[0])
    axes[0].set(xticks=range(13),yticks=range(13),xlabel='Predicted class (12 = no arrival)',ylabel='True class',title='Row-normalized confusion')
    for k in (3,6,12):
        prob=p[:,:k].sum(1);truth=y<k;predicted=[];observed=[]
        for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            mask=(prob>=a)&((prob<b) if b<1 else (prob<=b))
            if mask.any():predicted.append(prob[mask].mean());observed.append(truth[mask].mean())
        axes[1].plot(predicted,observed,'o-',label=f'Within {k*5} min')
    axes[1].plot([0,1],[0,1],'k--');axes[1].set(xlabel='Predicted probability',ylabel='Observed frequency',title='Reliability');axes[1].legend()
    plt.tight_layout();plt.show()


def plot_example(dataset,index,probabilities):
    x,y=dataset[index];fig,axes=plt.subplots(1,2,figsize=(13,4))
    # Only use for configurations containing radar as the first selected channel.
    axes[0].imshow(x[-1,0],vmin=0,vmax=1,cmap='viridis');center=x.shape[-1]//2
    axes[0].scatter(center,center,c='magenta',marker='+');axes[0].set_title('Latest source radar crop; target centered')
    axes[1].bar(range(13),probabilities);axes[1].set(xticks=range(13),xlabel='Arrival class (12 = no rain)',ylabel='Probability',title=f'Truth: {CLASSES[y]}')
    plt.tight_layout();plt.show()
