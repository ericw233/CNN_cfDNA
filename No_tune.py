import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import sys

from model import CNN
from load_data import load_data
from train_module import train_module

# ray tune
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from functools import partial

input_size = 60
feature_type = "Frag"

def test(data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch.csv",
         input_size=input_size,
         feature_type=feature_type):

    config = {
        "out1": 16,
        "out2": 64,
        "conv1": 2,
        "pool1": 2,
        "drop1": 0,
        
        "conv2": 2,
        "pool2": 2,
        "drop2": 0,
                                 
        "fc1": 256,
        "fc2": 64,
        "drop3": 0.5,
        
        "batch_size": 256
    }
    
    trained_model = train_module(config=config,data_dir=data_dir,input_size=input_size,feature_type=feature_type)
     
    

test()

