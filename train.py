#%% Import modules
import numpy as np
import torch

def train(A, ss, epoch, single_model, single_optim, loss_MSE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FR_ORDER_TRAIN = A["FR_ORDER_TRAIN"]
    POS_NOR_ORDER_TRAIN = A["POS_NOR_ORDER_TRAIN"]
    VEL_NOR_ORDER_TRAIN = A["VEL_NOR_ORDER_TRAIN"]
    ACC_NOR_ORDER_TRAIN = A["ACC_NOR_ORDER_TRAIN"]
    
    
    
    train_unsort_data = torch.from_numpy(FR_ORDER_TRAIN[ss]).type(torch.FloatTensor)
    train_label = np.concatenate((POS_NOR_ORDER_TRAIN[ss], VEL_NOR_ORDER_TRAIN[ss], ACC_NOR_ORDER_TRAIN[ss]), axis=2)
    train_label = torch.from_numpy(train_label).type(torch.FloatTensor)
    
    single_unsort_dataset = torch.utils.data.TensorDataset(train_unsort_data, train_label)
    single_unsort_dataloader = torch.utils.data.DataLoader(dataset = single_unsort_dataset, batch_size=32, shuffle=True)
    
    single_model.to(device)
    loss_MSE.to(device)
        
    # Training
    for ep in range(epoch):
        for n, (Data, Label) in enumerate(single_unsort_dataloader):
            single_optim.zero_grad()
        
            fr_data = Data
            valid_pos = Label[:, -1, :2]
            valid_vel = Label[:, -1, 2:4]
            valid_acc = Label[:, -1, 4:6]
        
        
            valid_pos = valid_pos.to(device)
            valid_vel = valid_vel.to(device)
            valid_acc = valid_acc.to(device)
            fr_data = fr_data.to(device)
        
            pred_pos, pred_vel, pred_acc = single_model(fr_data)
            loss_vel = loss_MSE(pred_vel, valid_vel)
            loss_acc = loss_MSE(pred_acc, valid_acc)
            loss_pos = loss_MSE(pred_pos, valid_pos)
            loss = loss_vel+ loss_acc + loss_pos
        
            loss.backward()
            single_optim.step()
            
        with torch.no_grad():
            print('epoch[{}], loss:{:.4f} >> pos loss:{:.4f}, vel loss:{:.4f}, acc loss:{:.4f}'
                  .format(ep+1, loss.item(), loss_pos.item(), loss_vel.item(), loss_acc.item()))
    
    torch.save(single_model, 'model.pth')