import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from copy import deepcopy
import os

from model import CNN_1D as CNN
from load_data import load_data_1D as load_data


class CNNwithTrainingTuning_1D(CNN):
    def __init__(self, config, input_size, num_class):
        model_config=self._match_params(config)                      # find the parameters for the original CNN class
        super(CNNwithTrainingTuning_1D, self).__init__(input_size, num_class, **model_config)        # pass the parameters into the original CNN class
        self.batch_size=config["batch_size"]
        self.num_epochs=config["num_epochs"]
        
        self.criterion=nn.BCELoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-4, weight_decay=1e-5)
        
    def _match_params(self, config):
        model_config={}
        model_keys=set(self.__annotations__.keys())

        for key, value in config.items():
            if key in model_keys:
                model_config[key] = value        
        return model_config
    
    def data_loader(self, data_dir, input_size, feature_type, R01BTuning):
        self.input_size=input_size
        self.feature_type=feature_type
        self.R01BTuning=R01BTuning
                    
        data, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_all_tensor, y_all_tensor, train_sampleid = load_data(data_dir, input_size, feature_type) 
        self.data_idonly=data[["SampleID","Train_Group"]]
        self.X_train_tensor=X_train_tensor
        self.y_train_tensor=y_train_tensor
        self.X_test_tensor=X_test_tensor
        self.y_test_tensor=y_test_tensor        
        self.X_all_tensor=X_all_tensor
        self.y_all_tensor=y_all_tensor
        self.train_sampleid=train_sampleid
        
        if(self.X_train_tensor.size(0) > 0):
            print("----- data loaded -----")
            print(f"Training frame has {self.X_train_tensor.size(0)} samples")
 
        if(R01BTuning==True):
            R01B_indexes=data.loc[data["Project"].isin(["R01BMatch"])].index
            self.X_train_tensor_R01B=self.X_all_tensor[R01B_indexes]
            self.y_train_tensor_R01B=self.y_all_tensor[R01B_indexes]
        
            if(self.X_train_tensor_R01B.size(0) > 0):
                print("----- R01B data loaded -----")
                print(f"R01B train frame has {self.X_train_tensor_R01B.size(0)} samples")
            
            
    def fit(self, output_path, R01BTuning_fit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)
               
        patience = 100  # Number of epochs with increasing test loss before early stopping
        min_test_loss = float("inf")  # Initialize minimum test loss
        max_test_auc = float(0.0)  # Initialize maximum test auc
        best_model = None  # Initialize test model
        epochs_without_improvement = 0  # Count of consecutive epochs without improvement

        for epoch in range(self.num_epochs):
            
            self.train()
            # Mini-batch training
            seed = 42 + epoch
            shuffled_indices = torch.randperm(self.X_train_tensor.size(0))
            self.X_train_tensor = self.X_train_tensor[shuffled_indices]
            self.y_train_tensor = self.y_train_tensor[shuffled_indices]
            
            for batch_start in range(0, len(self.X_train_tensor), self.batch_size):
                batch_end = batch_start + self.batch_size
                batch_X = self.X_train_tensor[batch_start:batch_end]
                batch_y = self.y_train_tensor[batch_start:batch_end]

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print the loss after every epoch
                train_auc = roc_auc_score(
                    batch_y.to('cpu').detach().numpy(), outputs.to('cpu').detach().numpy()
                )
                print(f"Epoch: {epoch+1}/{self.num_epochs}, i: {batch_start//self.batch_size}")
                print(f"Train Loss: {loss.item():.4f}, Train AUC: {train_auc.item():.4f}")
                print("-------------------------")

            # Evaluation on test data
            with torch.no_grad():
                self.eval()
                self.X_test_tensor=self.X_test_tensor.to(device)
                self.y_test_tensor=self.y_test_tensor.to(device)
                
                test_outputs=self(self.X_test_tensor)
                test_outputs=test_outputs.to("cpu")

                test_loss=self.criterion(test_outputs, self.y_test_tensor.to("cpu"))
                test_auc = roc_auc_score(self.y_test_tensor.to("cpu"), test_outputs.to("cpu"))
                print(f"Test Loss: {test_loss.item():.4f}, Test AUC: {test_auc.item():.4f}")
                print("-------------------------")

                # Early stopping check
                if test_auc >= max_test_auc:
                    max_test_auc = test_auc
                    best_model = deepcopy(self.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered! No improvement in {patience} epochs.")
                        break
        self.max_test_auc = max_test_auc
        self.load_state_dict(best_model)
        
        # export the best model
        if not os.path.exists(f"{output_path}/Raw/"):
            os.makedirs(f"{output_path}/Raw/")
        torch.save(self.state_dict(),f"{output_path}/Raw/{self.feature_type}_CNN_best.pt")
        
        # obtain scores of all samples and export
        with torch.no_grad():
            self.eval()

            self.X_all_tensor = self.X_all_tensor.to(device)
            outputs_all = self(self.X_all_tensor)
            outputs_all = outputs_all.to("cpu")

            self.data_idonly['CNN_score'] = outputs_all.detach().cpu().numpy()
            self.data_idonly.to_csv(f"{output_path}/Raw/{self.feature_type}_score.csv", index=False)
                
        # fine tuning with R01BMatch data
        if(self.R01BTuning and R01BTuning_fit):
            self.train()
            optimizer_R01B = torch.optim.Adam(self.parameters(), lr=1e-6)

            # Perform forward pass and compute loss
            self.X_train_tensor_R01B = self.X_train_tensor_R01B.to(device)
            self.y_train_tensor_R01B = self.y_train_tensor_R01B.to(device)

            for epoch_toupdate in range(30):
                outputs_R01B = self(self.X_train_tensor_R01B)
                loss = self.criterion(outputs_R01B, self.y_train_tensor_R01B)

                # Backpropagation and parameter update
                optimizer_R01B.zero_grad()
                loss.backward()
                optimizer_R01B.step()
            
            if not os.path.exists(f"{output_path}/R01BTuned/"):
                os.makedirs(f"{output_path}/R01BTuned/")    
            torch.save(self.state_dict(),f"{output_path}/R01BTuned/{self.feature_type}_CNN_best_R01BTuned.pt")
            
            with torch.no_grad():
                self.eval()
                
                self.X_test_tensor=self.X_test_tensor.to(device)
                self.y_test_tensor=self.y_test_tensor.to(device)
                
                test_outputs=self(self.X_test_tensor)
                test_outputs=test_outputs.to("cpu")

                test_loss=self.criterion(test_outputs, self.y_test_tensor.to("cpu"))
                test_auc = roc_auc_score(self.y_test_tensor.to("cpu"), test_outputs.to("cpu"))
                print(f"Test Loss (tuned): {test_loss.item():.4f}, Test AUC (tuned): {test_auc.item():.4f}")
                print("-------------------------")
                
                ### obtain scores of all samples
                self.X_all_tensor = self.X_all_tensor.to(device)
                outputs_all_tuned = self(self.X_all_tensor)
                outputs_all_tuned = outputs_all_tuned.to("cpu")

            self.data_idonly['CNN_score_tuned'] = outputs_all_tuned.detach().cpu().numpy()
            self.data_idonly.to_csv(f"{output_path}/R01BTuned/{self.feature_type}_score_R01BTuned.csv", index=False)
    
    
    def predict(self, X_predict_tensor, y_predict_tensor):
        
        X_predict_tensor = X_predict_tensor.to(self.device)
        y_predict_tensor = y_predict_tensor.to(self.device)
        with torch.no_grad():
            self.eval()
            outputs_predict = self(X_predict_tensor)        
        return(outputs_predict.detach().cpu().numpy())
            
                        
                    
        
        
