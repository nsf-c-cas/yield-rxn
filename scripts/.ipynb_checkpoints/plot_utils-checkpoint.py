#!/usr/bin/env python3

"""
This script contains utility functions used in 
05_plots.ipynb

"""

import matplotlib.pyplot as plt
from collections import defaultdict
from scripts import plot_utils as pu

import numpy as np

import pickle
import os
import pandas as pd


def plot_actual_vs_predicted(model_results,fig_path,alpha):
    y_true = model_results['yield'].values
    y_pred = model_results['pred_yield'].values

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_true, y_pred, color='b',alpha=alpha)
    ax.set_xlabel('Actual yields-'+key,fontsize=15)
    ax.set_ylabel('Predicted yields-'+key,fontsize=15)
    
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    plt.tight_layout()

    plt.savefig(fig_path+'actual_vs_predicted.png',dpi=300)
    plt.show()
    
def plot_fig(w1,w2,xlim, ylim,fig_path):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(w1, label="Domain weight")
    ax.plot(w2, label="Graph weight")
    
    ax.set_xlabel('Number of epochs',fontsize=15)
    ax.set_ylabel('Weight value',fontsize=15)
    ax.set_xlim([0,xlim])
   
    plt.legend(fontsize=14,bbox_to_anchor=(0.9, -0.2),ncol=2,shadow=True)
    plt.tight_layout()
    plt.savefig(fig_path+'weights_curve.png',dpi=300)
    plt.show()
    
def plot_fig_2(train_scores,test_scores,model_path):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(6,4))
    
    ax.plot(train_scores, label="Test curve")
    ax.plot(test_scores, label="Train curve")
    
    ax.set_xlabel('Number of epochs',fontsize=15)
    ax.set_ylabel('R^2',fontsize=15)
    ax.set_ylim([0,1])

    #ax.legend(loc="center right")
    #plt.tight_layout()
    
    plt.savefig(model_path+'train_curve.png',dpi=300)
    plt.show()