# -*- coding: utf-8 -*-
"""BOHB_XGB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OMYoUJpyud1sB-jrwTmQwj4QdgYvUhpE
"""

import requests
import os
import hashlib
import tempfile
import gzip

import torch
import numpy as np

from bohb import BOHB
import bohb.configspace as cs

from sklearn.ensemble import GradientBoostingClassifier

SEED = 123


if __name__ == '__main__':
    SEED=7
    
    fish=pd.read_csv('Fish.csv')
    np.random.seed(SEED)
    fish['Species']=fish['Species'].map({'Perch':0,'Bream':1,'Roach':2,'Pike':3,'Parkki':4,'Whitefish':5,'Smelt':6})
    dev_frac=0.8


    n_obs=fish.shape[0]
    n_dev=round(n_obs*dev_frac)

    dev=fish.iloc[:n_dev]
    oot=fish.iloc[n_dev:]
    dev['Species'][12]=3
    target='Species'

    max_budget=2000
    min_budget=1
    


    #### Run Algo
    from bohb.setup import train_xgb

    def worker(params, budget,max_budget=max_budget, min_budget=min_budget):
        subsample=(((1-0.05)/(max_budget-min_budget))*(budget-min_budget))+0.05
        # subsample=(1-1/budget**2)
        loss = train_xgb(**params, dev=dev, oot=oot,target=target, subsample=subsample)
        return loss

    n_trees = cs.IntegerUniformHyperparameter('n_trees', lower=1,upper=20)
    max_depth = cs.IntegerUniformHyperparameter('max_depth', lower=1,upper=8)

    # optimizer = cs.CategoricalHyperparameter('optimizer', ['adam', 'sgd', 'rms'])
    learning_rate = cs.UniformHyperparameter('learning_rate', 1e-4, 1e-1, log=True)

    configspace = cs.ConfigurationSpace([n_trees, max_depth, learning_rate],
                                        seed=7)

    opt = BOHB(configspace, worker, max_budget=max_budget, min_budget=min_budget, n_proc=2)
    logs = opt.optimize()
    print(logs)
