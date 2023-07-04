import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import sys

# ray tune package-related functions
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from functools import partial

### self defined functions
from model import CNN
from load_data import load_data
from ray_tune import ray_tune
from train_and_tune import CNNwithTrainingTuning
from cross_validation import CNNwithCV

# default value of input_size and feature_type
feature_type = "Griffin"
input_size = 55
tuning_num = 200
epoch_num = 1000
output_path="/mnt/binf/eric/CNN_Mercury_testoutput_0630"

# get argument values from external inputs
if len(sys.argv) >= 4:
    feature_type = sys.argv[1]
    input_size = int(sys.argv[2])
    tuning_num = int(sys.argv[3])
    epoch_num = int(sys.argv[4])
    output_path = sys.argv[5]
    print(f"Getting arguments: feature type: {feature_type}, input size: {input_size}, \
        tuning round: {tuning_num}, epoch num: {epoch_num}, output path: {output_path}\n")  
else:
    print(f"Not enough inputs, using default arguments: feature type: {feature_type}, input size: {input_size}, \
        tuning round: {tuning_num}, epoch num: {epoch_num}, output path: {output_path}\n")

num_class = 2
output_path_cv=f"{output_path}_cv"
data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch.csv"

best_config={"out1": 16,"out2": 64,"conv1": 2,"pool1": 2,"drop1": 0,"conv2": 2,"pool2": 2,"drop2": 0,
             "fc1": 256,"fc2": 64,"drop3": 0.5,"batch_size": 256,"num_epochs": 2000}

try:
    best_config, best_testloss=ray_tune(num_samples=tuning_num, 
                                max_num_epochs=epoch_num, 
                                gpus_per_trial=1,
                                output_path=output_path,
                                data_dir=data_dir,
                                input_size=input_size,
                                feature_type=feature_type)
except Exception as e:
    print("==========   Ray tune failed! An error occurred:", str(e),"   ==========")

print("***********************   Ray tune finished   ***********************************")

# print(best_config)

###### load best_config from text
# import ast
# # Specify the path to the text file
# config_file = '/mnt/binf/eric/CNN_Mercury_output/_config.txt'

# with open(config_file, 'r') as cf:
#     config_txt = cf.read()
# config_dict = ast.literal_eval(config_txt)
# # Print the dictionary
# print(config_dict)

# best_config=config_dict
# best_config["num_epochs"]=2000

#### train and tune CNN; CNNwithTrainingTuning class takes all variables in the config dictionary from ray_tune
print("***********************   Start fittiing model with best configurations   ***********************************")
CNN_trainvalid=CNNwithTrainingTuning(best_config, input_size=input_size,num_class=num_class)
CNN_trainvalid.data_loader(data_dir=data_dir,
                     input_size=input_size,
                     feature_type=feature_type,
                     R01BTuning=True)

CNN_trainvalid.fit(output_path=output_path,
             R01BTuning_fit=True)

print("***********************   Completed fitting model   ***********************************")

#### cv process is independent
print("***********************   Start cross validations   ***********************************")
CNN_cv=CNNwithCV(best_config, input_size=input_size,num_class=num_class)
CNN_cv.data_loader(data_dir=data_dir,
                    input_size=input_size,
                    feature_type=feature_type,
                    R01BTuning=True)

CNN_cv.crossvalidation(output_path=output_path_cv,num_folds=5,R01BTuning_fit=True)
print("***********************   Completed cross validations   ***********************************")


