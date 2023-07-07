import torch
import torch.nn as nn
# Define the CNN model
# CNN model
class CNN(nn.Module):
    def __init__(self, input_size=25, num_class=2, out1=16, out2=64, conv1=2, pool1=2, drop1=0.1, conv2=2, pool2=2, drop2=0.1, fc1=256, fc2=64, drop3=0.5):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=None)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out1)
        self.dropout1 = nn.Dropout(drop1)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=conv2, stride=2, bias=None)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out2)
        self.dropout2 = nn.Dropout(drop2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=out2, out_channels=256, kernel_size=2, stride=2, bias=None)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        # Calculate the input size for the fully connected layer based on the output size after convolutions and pooling
        self.fc_input_size = self._get_fc_input_size(input_size)
        self.fc1 = nn.Linear(self.fc_input_size, fc1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(drop3)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size, input_size)
        x = self.conv1(dummy_input)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        # x = self.pool3(x)
        
        flattened_size = x.size(1) * x.size(2) * x.size(3)
        return flattened_size
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        # x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x.squeeze(1)


class CNN_1D(nn.Module):
    def __init__(self, input_size=1000, num_class=2, out1=16, out2=64, conv1=2, pool1=2, drop1=0.1, conv2=2, pool2=2, drop2=0.1, fc1=256, fc2=64, drop3=0.5):
        super(CNN_1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=None)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out1)
        self.dropout1 = nn.Dropout(drop1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool1, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=conv2, stride=2, bias=None)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(out2)
        self.dropout2 = nn.Dropout(drop2)
        self.pool2 = nn.MaxPool1d(kernel_size=pool2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=out2, out_channels=256, kernel_size=2, stride=2, bias=None)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
                
        # Calculate the input size for the fully connected layer based on the output size after convolutions and pooling
        self.fc_input_size = self._get_fc_input_size(input_size)
        self.fc1 = nn.Linear(self.fc_input_size, fc1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(drop3)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.conv1(dummy_input)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        # x = self.pool3(x)
        
        flattened_size = x.size(1) * x.size(2)
        return flattened_size
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        # x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x.squeeze(1)
