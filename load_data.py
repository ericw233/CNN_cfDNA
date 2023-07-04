import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pad_and_reshape import pad_and_reshape
from sklearn.preprocessing import MinMaxScaler

def load_data(data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch.csv", input_size=32, feature_type = "Arm"):
    # Read data from CSV file
    data = pd.read_csv(data_dir).dropna(axis=1)

    # keep a full dataset without shuffling
    mapping = {'Healthy':0,'Cancer':1}
    X_all = data.filter(regex = feature_type, axis=1).dropna(axis=1)
    y_all = data.loc[:,'Train_Group'].replace(mapping)

    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = data.loc[data["train"] == "training"].filter(regex = feature_type, axis=1)
    y_train = data.loc[data["train"] == "training","Train_Group"].replace(mapping)

    X_test = data.loc[data["train"] == "validation"].filter(regex = feature_type, axis=1)
    y_test = data.loc[data["train"] == "validation","Train_Group"].replace(mapping)

    # Scale the features to a suitable range (e.g., [0, 1])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_all)

    # Convert the data to PyTorch tensors
    input_size = input_size
    X_train_tensor = pad_and_reshape(X_train_scaled, input_size).type(torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = pad_and_reshape(X_test_scaled, input_size).type(torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    X_all_tensor = pad_and_reshape(X_all_scaled, input_size).type(torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)

    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values

    return data, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_all_tensor, y_all_tensor, train_sampleid


### copy and paste from load_data, not using pad_and_reshape, keeping the original 1D dimensions
def load_data_1D(data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch.csv", input_size=32, feature_type = "Arm"):
    # Read data from CSV file
    data = pd.read_csv(data_dir).dropna(axis=1)

    # keep a full dataset without shuffling
    mapping = {'Healthy':0,'Cancer':1}
    X_all = data.filter(regex = feature_type, axis=1).dropna(axis=1)
    y_all = data.loc[:,'Train_Group'].replace(mapping)

    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = data.loc[data["train"] == "training"].filter(regex = feature_type, axis=1)
    y_train = data.loc[data["train"] == "training","Train_Group"].replace(mapping)

    X_test = data.loc[data["train"] == "validation"].filter(regex = feature_type, axis=1)
    y_test = data.loc[data["train"] == "validation","Train_Group"].replace(mapping)

    # Scale the features to a suitable range (e.g., [0, 1])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_all)

    # Convert the data to PyTorch tensors
    input_size = input_size
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)

    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values

    return data, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_all_tensor, y_all_tensor, train_sampleid

