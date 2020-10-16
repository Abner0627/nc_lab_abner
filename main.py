#%% Import modules
import time
from get_data import get_data
import model as m
from train import train
from comp import comp
import numpy as np
from os import listdir
from sklearn.metrics import r2_score
import torch
import torch.optim as optim
import torch.nn as nn


#%%
tStart = time.time()

#%% Path
dpath = './Data_I'
data_list = listdir(dpath)

#%% Selection session
single_session = bool(0)
# 1為讀取單一session；0為讀取多個session的資料

if single_session:    
# 讀取單一session
    session = 0    
    # 欲讀取session的編號，從0開始，共37個session
    data_list = data_list[session:session+1]
else:    
# 讀取多個session
    session_start = 0
    # 讀取
    session_end = 36
    data_list = data_list[session_start:session_end+1]

data_list.sort()  
print('\n---------------')  
print('Num of session: ', len(data_list))    

#%% Get data
print('\nGet data...')
DATA, NOR = get_data(dpath, data_list)

#%% Model and parameters
model = m.CNN()
Epoch = 30
lr = 1e-4    # Learning rate
single_optim = optim.Adam(model.parameters(), lr=lr)    # Optimizer
loss_MSE = nn.MSELoss()    # Loss function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% For test
FR_ORDER_TEST = DATA["FR_ORDER_TEST"]
POS_TEST = DATA["POS_TEST"]
VEL_TEST = DATA["VEL_TEST"]
ACC_TEST = DATA["ACC_TEST"]

POS_MAX = NOR["POS_MAX"]
POS_MIN = NOR["POS_MIN"]
POS_MEAN = NOR["POS_MEAN"]
VEL_MAX = NOR["VEL_MAX"]
VEL_MIN = NOR["VEL_MIN"]
VEL_MEAN = NOR["VEL_MEAN"]
ACC_MAX = NOR["ACC_MAX"]
ACC_MIN = NOR["ACC_MIN"]
ACC_MEAN = NOR["ACC_MEAN"]

Pred_POS_X = []
Pred_POS_Y = []
Pred_VEL_X = []
Pred_VEL_Y = []
Pred_ACC_X = []
Pred_ACC_Y = []

R2_POS_X = []
R2_POS_Y = []
R2_VEL_X = []
R2_VEL_Y = []
R2_ACC_X = []
R2_ACC_Y = []

#%% Training
for ss in range(len(data_list)):
    single_model = model
    print('\nSession_' + str(ss+1))
    print('-----Training-----')
    train(DATA, ss, Epoch, single_model, single_optim, loss_MSE)

#%% Testing    
    print('-----Testing-----')
    test_unsort_data = torch.from_numpy(FR_ORDER_TEST[ss]).type(torch.FloatTensor)
    test_label = np.concatenate((POS_TEST[ss], VEL_TEST[ss], ACC_TEST[ss]), axis=1)
    test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
        
    single_test_dataset = torch.utils.data.TensorDataset(test_unsort_data, test_label)
    single_test_dataloader = torch.utils.data.DataLoader(dataset = single_test_dataset, batch_size=32, shuffle=False)
    
    model_train = torch.load('model.pth')
    
    with torch.no_grad():
        for n_ts, (Data_ts, Label_ts) in enumerate (single_test_dataloader):
    
            fr_data = Data_ts
            fr_data = fr_data.to(device)
                
            out_pos, out_vel, out_acc = model_train(fr_data)
    
            out_vel = out_vel.cpu().data.numpy()
            out_acc = out_acc.cpu().data.numpy()
            out_pos = out_pos.cpu().data.numpy()
            
            if n_ts == 0:
                pred_vel = out_vel
                pred_acc = out_acc
                pred_pos = out_pos
    
            else:
                pred_vel = np.concatenate((pred_vel, out_vel), axis=0)
                pred_acc = np.concatenate((pred_acc, out_acc), axis=0)
                pred_pos = np.concatenate((pred_pos, out_pos), axis=0)
                
        # R2 and Pred
        pred_px , pred_py = pred_pos[:, 0], pred_pos[:, 1]
        real_px , real_py = POS_TEST[ss][:, 0], POS_TEST[ss][:, 1] 
        pred_vx , pred_vy = pred_vel[:, 0], pred_vel[:, 1]
        real_vx , real_vy = VEL_TEST[ss][:, 0], VEL_TEST[ss][:, 1]
        pred_ax , pred_ay = pred_acc[:, 0], pred_acc[:, 1]
        real_ax , real_ay = ACC_TEST[ss][:, 0], ACC_TEST[ss][:, 1]
        
        Pred_POS_X.append(pred_px *(POS_MAX[ss][0,0] - POS_MIN[ss][0,0]) + POS_MEAN[ss][0,0])
        Pred_POS_Y.append(pred_py *(POS_MAX[ss][0,1] - POS_MIN[ss][0,1]) + POS_MEAN[ss][0,1])
        Pred_VEL_X.append(pred_vx *(VEL_MAX[ss][0,0] - VEL_MIN[ss][0,0]) + VEL_MEAN[ss][0,0])
        Pred_VEL_Y.append(pred_vy *(VEL_MAX[ss][0,1] - VEL_MIN[ss][0,1]) + VEL_MEAN[ss][0,1])
        Pred_ACC_X.append(pred_ax *(ACC_MAX[ss][0,0] - ACC_MIN[ss][0,0]) + ACC_MEAN[ss][0,0])
        Pred_ACC_Y.append(pred_ay *(ACC_MAX[ss][0,1] - ACC_MIN[ss][0,1]) + ACC_MEAN[ss][0,1])
                    
        print('r2_px :', np.round(r2_score(real_px, Pred_POS_X[ss]),4))
        print('r2_py :', np.round(r2_score(real_py, Pred_POS_Y[ss]),4))
        
        print('r2_vx :', np.round(r2_score(real_vx, Pred_VEL_X[ss]),4))
        print('r2_vy :', np.round(r2_score(real_vy, Pred_VEL_Y[ss]),4))
                
        print('r2_ax :', np.round(r2_score(real_ax, Pred_ACC_X[ss]),4))
        print('r2_ay :', np.round(r2_score(real_ay, Pred_ACC_Y[ss]),4))
              
        R2_POS_X.append(np.round(r2_score(real_px, Pred_POS_X[ss]),4))
        R2_POS_Y.append(np.round(r2_score(real_py, Pred_POS_Y[ss]),4))
        R2_VEL_X.append(np.round(r2_score(real_vx, Pred_VEL_X[ss]),4))
        R2_VEL_Y.append(np.round(r2_score(real_vy, Pred_VEL_Y[ss]),4))
        R2_ACC_X.append(np.round(r2_score(real_ax, Pred_ACC_X[ss]),4))
        R2_ACC_Y.append(np.round(r2_score(real_ay, Pred_ACC_Y[ss]),4))
            
        with open('R2_CNN.npy', 'wb') as f:
            np.save(f, R2_POS_X)
            np.save(f, R2_POS_Y)
            np.save(f, R2_VEL_X)
            np.save(f, R2_VEL_Y)
            np.save(f, R2_ACC_X)
            np.save(f, R2_ACC_Y)

#%% Compare
# =============================================================================
# with open('R2_CNN.npy', 'rb') as f:
#     R2_POS_X = np.load(f)
#     R2_POS_Y = np.load(f)
#     R2_VEL_X = np.load(f)
#     R2_VEL_Y = np.load(f)
#     R2_ACC_X = np.load(f)
#     R2_ACC_Y = np.load(f)
# =============================================================================
        
if len(data_list) != 37:
    print("Only used in all sessions")
else:
    comp(R2_POS_X, R2_POS_Y, R2_VEL_X, R2_VEL_Y, R2_ACC_X, R2_ACC_Y)
    
#%%
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))