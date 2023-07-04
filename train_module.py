import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from ray.air import Checkpoint, session
from model import CNN
from load_data import load_data

def train_module(config, data_dir, input_size, feature_type):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = CNN(input_size=input_size, num_class=2, 
                out1=config["out1"], out2=config["out2"], 
                conv1=config["conv1"], pool1=config["pool1"], drop1=config["drop1"], 
                conv2=config["conv2"], pool2=config["pool2"], drop2=config["drop2"], 
                fc1=config["fc1"], fc2=config["fc2"], drop3=config["drop3"])
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Get checkpoint
    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Load data using load_data()
    _, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_all_tensor, y_all_tensor, _ = load_data(data_dir, input_size, feature_type) 
    
    # Training loop
    num_epochs = int(config["num_epochs"])
    batch_size = int(config["batch_size"])
    patience = 100  # Number of epochs with increasing test loss before early stopping
    min_test_loss = float("inf")  # Initialize minimum test loss
    max_test_auc = float(0.0)  # Initialize maximum test auc
    best_model = None  # Initialize test model state dict
    epochs_without_improvement = 0  # Count of consecutive epochs without improvement

    for epoch in range(num_epochs):
        
        model.train()
        # Mini-batch training
        seed = 42 + epoch
        shuffled_indices = torch.randperm(X_train_tensor.size(0))
        X_train_tensor = X_train_tensor[shuffled_indices]
        y_train_tensor = y_train_tensor[shuffled_indices]
        
        for batch_start in range(0, len(X_train_tensor), batch_size):
            batch_end = batch_start + batch_size
            batch_X = X_train_tensor[batch_start:batch_end]
            batch_y = y_train_tensor[batch_start:batch_end]

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Print the loss after every epoch
            train_auc = roc_auc_score(
                batch_y.to('cpu').detach().numpy(), outputs.to('cpu').detach().numpy()
            )
            print(f"Epoch: {epoch+1}/{num_epochs}, i: {batch_start//batch_size}, Train Loss: {loss.item():.4f}, Train AUC: {train_auc.item():.4f}")
            print("-------------------------")

        # Evaluation on test data
        with torch.no_grad():
            model.eval()
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            test_outputs = model(X_test_tensor)
            test_outputs = test_outputs.to("cpu")

            test_loss = criterion(test_outputs, y_test_tensor.to("cpu"))
            test_auc = roc_auc_score(y_test_tensor.to("cpu"), test_outputs.to("cpu"))
            print(f"Test Loss: {test_loss.item():.4f}, Test AUC: {test_auc.item():.4f}")
            print("-------------------------")

            # Early stopping check
            if test_auc >= max_test_auc:
                max_test_auc = test_auc
                best_model = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered! No improvement in {patience} epochs.")
                    break
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)
        
        session.report(
            {"testloss": float(test_loss.item()), "testauc": test_auc},
            checkpoint=checkpoint,
        )
        
    model.load_state_dict(best_model)
    # torch.save(model,f"/mnt/binf/eric/CNN_2D_RayTune/{feature_type}_CNN_2D_RayTune.pt")
    print("Training module complete")
    # return(model)