import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from copy import deepcopy

from model import CNN
from load_data import load_data   
    
class CNNwithCV(CNN):
    def __init__(self, config, input_size, num_class):
        model_config=self._match_params(config)
        super(CNNwithCV, self).__init__(input_size, num_class, **model_config)
        self.batch_size=config["batch_size"]
        self.num_epochs=config["num_epochs"]
        
        self.criterion=nn.BCELoss()
        self.optimizer=torch.optim.Adam(self.parameters())
        
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
        
        if(R01BTuning==True):
            R01B_indexes=data.loc[data["Project"].isin(["R01BMatch"])].index
            self.X_train_tensor_R01B=X_all_tensor[R01B_indexes]
            self.y_train_tensor_R01B=y_all_tensor[R01B_indexes]
    
    def weight_reset(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.reset_parameters()
        
    def crossvalidation(self,num_folds, output_path, R01BTuning_fit):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kf = KFold(n_splits=num_folds, shuffle=True)
    
        fold_scores = []  # List to store validation scores
        fold_labels = []
        fold_numbers = []
        fold_sampleid = []
        fold_scores_tuned = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(self.X_train_tensor)):
            X_train_fold, X_val_fold = self.X_train_tensor[train_index], self.X_train_tensor[val_index]
            y_train_fold, y_val_fold = self.y_train_tensor[train_index], self.y_train_tensor[val_index]
            sampleid_val_fold = self.train_sampleid[val_index]
            
            ### reset the model
            self.weight_reset()
            self.to(device)
                        
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
            optimizer_tuned = torch.optim.Adam(self.parameters(), lr=1e-6)
            patience = 100
            max_test_auc = 0.0
            best_model_cv = None
            epochs_without_improvement = 0
            
            for epoch in range(self.num_epochs):
                shuffled_indices = torch.randperm(X_train_fold.shape[0])
                X_train_fold = X_train_fold[shuffled_indices]
                y_train_fold = y_train_fold[shuffled_indices]
                
                ### turn to train mode
                self.train()
                
                for batch_start in range(0, X_train_fold.shape[0], self.batch_size):
                    batch_end = batch_start + self.batch_size
                    batch_X = X_train_fold[batch_start:batch_end].to(device)
                    batch_y = y_train_fold[batch_start:batch_end].to(device)
        
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        
                    train_auc = roc_auc_score(
                        batch_y.to('cpu').detach().numpy(), outputs.to('cpu').detach().numpy()
                    )
                    print(f"Fold: {fold+1}/{num_folds}, Epoch: {epoch+1}/{self.num_epochs}, i: {batch_start//self.batch_size}")
                    print(f"Train Loss: {loss.item():.4f}, Train AUC: {train_auc.item():.4f}")
                    print("-------------------------")
        
                with torch.no_grad():
                    self.eval()
                    val_outputs = self(X_val_fold.to(device))
                    val_outputs = val_outputs.to("cpu")
        
                    val_loss = criterion(val_outputs.to("cpu"), y_val_fold.to("cpu"))
                    val_auc = roc_auc_score(y_val_fold.to("cpu"), val_outputs.to("cpu"))
                    print(f"Fold {fold+1}/{num_folds}, Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss.item():.4f}, Validation AUC: {val_auc.item():.4f}")
                    print("-------------------------")
        
                    if val_auc >= max_test_auc:
                        max_test_auc = val_auc
                        best_model_cv = deepcopy(self.state_dict())
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            print(f"Early stopping triggered for Fold {fold+1}! No improvement in {patience} epochs.")
                            break
            
            self.load_state_dict(best_model_cv)
            
            if not os.path.exists(f"{output_path}/Raw/"):
                os.makedirs(f"{output_path}/Raw/")
             
            torch.save(self.state_dict(), f"{output_path}/Raw/{self.feature_type}_CNN_cv_fold{fold+1}.pt")
            fold_scores.append(val_outputs.detach().cpu().numpy())  # Collect validation scores for the fold
            fold_labels.append(y_val_fold.detach().cpu().numpy())
            fold_numbers.append(np.repeat(fold+1, len(y_val_fold.detach().cpu().numpy())))
            fold_sampleid.append(sampleid_val_fold)
            
            # add model tuning with R01B
            self.train()
            for epoch_tuned in range(30):
                self.X_train_tensor_R01B = self.X_train_tensor_R01B.to(device)
                self.y_train_tensor_R01B = self.y_train_tensor_R01B.to(device)
                
                optimizer_tuned.zero_grad()
                outputs_tuned = self(self.X_train_tensor_R01B)
                loss = criterion(outputs_tuned, self.y_train_tensor_R01B)
                loss.backward()
                optimizer_tuned.step()
            
            if not os.path.exists(f"{output_path}/R01BTuned/"):
                os.makedirs(f"{output_path}/R01BTuned/")           
            torch.save(self.state_dict(), f"{output_path}/R01BTuned/{self.feature_type}_CNN_cv_fold{fold+1}_R01Btuned.pt")
                    
            # results of tuned model
            with torch.no_grad():
                self.eval()
                val_outputs = self(X_val_fold.to(device))
                val_outputs = val_outputs.to("cpu")
        
                val_loss = criterion(val_outputs.to("cpu"), y_val_fold.to("cpu"))
                val_auc = roc_auc_score(y_val_fold.to("cpu"), val_outputs.to("cpu"))
                print(f"Fold {fold+1}/{num_folds}, Epoch {epoch+1}/{self.num_epochs}, Validation Loss (Tuned): {val_loss.item():.4f}, Validation AUC (Tuned): {val_auc.item():.4f}")
                print("-------------------------")          
                
            fold_scores_tuned.append(val_outputs.detach().cpu().numpy())  # Collect validation scores for the fold
                    
        all_scores = np.concatenate(fold_scores)
        all_labels = np.concatenate(fold_labels)
        all_numbers = np.concatenate(fold_numbers)
        all_sampleid = np.concatenate(fold_sampleid)
        all_scores_tuned = np.concatenate(fold_scores_tuned)

        
        # Save fold scores to CSV file
        df = pd.DataFrame({'Fold': all_numbers,
                        'Scores': all_scores,
                        'Scores_tuned': all_scores_tuned,
                        'Train_Group': all_labels,
                        'SampleID': all_sampleid})
        
        df.to_csv(f"{output_path}/{self.feature_type}_CV_score.csv", index=False)