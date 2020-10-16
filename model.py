#%%
import torch
import torch.nn as nn

#%% Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv_pos = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(20,3), padding=(0,1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size = (1, 9), dilation = (1, 9)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size = (1, 9)),
            nn.BatchNorm2d(128),
            nn.PReLU()
            )
            
        self.Conv_vel = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(20,3), padding=(0,1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size = (1, 9), dilation = (1, 9)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size = (1, 9)),
            nn.BatchNorm2d(128),
            nn.PReLU()
            )
        
        self.Conv_acc = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(20,3), padding=(0,1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size = (1, 9), dilation = (1, 9)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size = (1, 9)),
            nn.BatchNorm2d(128),
            nn.PReLU()
            )        
        
        self.FC_pos = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
            )
            
        self.FC_vel = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
            )
            
        self.FC_acc = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
            )
        
    
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x_conv_pos = self.Conv_pos(x)
        x_conv_vel = self.Conv_vel(x)
        x_conv_acc = self.Conv_acc(x)
        x_reshape_pos = x_conv_pos.view(-1, 128*4*4)
        x_reshape_vel = x_conv_vel.view(-1, 128*4*4)
        x_reshape_acc = x_conv_acc.view(-1, 128*4*4)
        
        pred_pos = self.FC_pos(x_reshape_pos)
        pred_vel = self.FC_pos(x_reshape_vel)
        pred_acc = self.FC_pos(x_reshape_acc)
        
        return pred_pos, pred_vel, pred_acc
        
        
        