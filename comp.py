#%% Import modules
import numpy as np
import matplotlib.pyplot as plt

#%%
def comp(R2_POS_X, R2_POS_Y, R2_VEL_X, R2_VEL_Y, R2_ACC_X, R2_ACC_Y):
    px = np.hstack((np.array(R2_POS_X), np.mean(np.array(R2_POS_X))))
    py = np.hstack((np.array(R2_POS_Y), np.mean(np.array(R2_POS_Y))))
    vx = np.hstack((np.array(R2_VEL_X), np.mean(np.array(R2_VEL_X))))
    vy = np.hstack((np.array(R2_VEL_Y), np.mean(np.array(R2_VEL_Y))))
    ax = np.hstack((np.array(R2_ACC_X), np.mean(np.array(R2_ACC_X))))
    ay = np.hstack((np.array(R2_ACC_Y), np.mean(np.array(R2_ACC_Y))))
    
    #Plot
    x_ticks = []
    width = 0.15
    for n in range(38):
        x_ticks.append(n+1)
    x_ticks = np.array(x_ticks)
    y_ticks= np.array([-0.2,0, 0.2, 0.4, 0.6, 0.8,1,1.2])
    
    fig, ax_= plt.subplots(2, 1, figsize = (20,12))
    ax_[0].set_title('CNN', size=20 )
    ax_[0].bar(x_ticks, px, width, label='CNN', color='teal')
    ax_[0].set_ylabel('Position R$^{2}$ X', size=25)
    ax_[0].set_yticks(y_ticks)
    ax_[0].set_yticklabels(y_ticks, size = 20)
    ax_[0].set_xticks(x_ticks)
    ax_[0].set_xticklabels(x_ticks, size=20)
    ax_[0].legend(fontsize = 20, loc=3) 
    for a,b in zip(x_ticks, px):
        ax_[0].text(a, b+0.05, '{:.2f}'.format(b), ha='center', va='bottom', fontsize=10)
    
    ax_[1].bar(x_ticks, py, width, label='CNN', color='teal')
    ax_[1].set_ylabel('Position R$^{2}$ Y', size=25)
    ax_[1].set_yticks(y_ticks)
    ax_[1].set_yticklabels(y_ticks, size = 20)
    ax_[1].set_xticks(x_ticks)
    ax_[1].set_xticklabels(x_ticks, size=20)
    ax_[1].legend(fontsize = 20, loc=3)
    for a,b in zip(x_ticks, py):
        ax_[1].text(a, b+0.05, '{:.2f}'.format(b), ha='center', va='bottom', fontsize=10)    
    
    plt.tight_layout()
    plt.savefig('Compare_pos.png')
    
    fig, ax_= plt.subplots(2, 1, figsize = (20,12))
    ax_[0].bar(x_ticks, vx, width, label='CNN', color='teal')
    ax_[0].set_ylabel('Velocity R$^{2}$ X', size=25)
    ax_[0].set_yticks(y_ticks)
    ax_[0].set_yticklabels(y_ticks, size = 20)
    ax_[0].set_xticks(x_ticks)
    ax_[0].set_xticklabels(x_ticks, size=20)
    ax_[0].legend(fontsize = 20, loc=3)
    for a,b in zip(x_ticks, vx):
        ax_[0].text(a, b+0.05, '{:.2f}'.format(b), ha='center', va='bottom', fontsize=10)    
    
    ax_[1].bar(x_ticks, vy, width, label='CNN', color='teal')
    ax_[1].set_ylabel('Velocity R$^{2}$ Y', size=25)
    ax_[1].set_yticks(y_ticks)
    ax_[1].set_yticklabels(y_ticks, size = 20)
    ax_[1].set_xticks(x_ticks)
    ax_[1].set_xticklabels(x_ticks, size=20)
    ax_[1].legend(fontsize = 20, loc=3)
    for a,b in zip(x_ticks, vy):
        ax_[1].text(a, b+0.05, '{:.2f}'.format(b), ha='center', va='bottom', fontsize=10)    
    
    plt.tight_layout()
    plt.savefig('Compare_vel.png')
    
    fig, ax_= plt.subplots(2, 1, figsize = (20,12))
    ax_[0].bar(x_ticks, ax, width, label='CNN', color='teal')
    ax_[0].set_ylabel('Accelerate R$^{2}$ X', size=25)
    ax_[0].set_yticks(y_ticks)
    ax_[0].set_yticklabels(y_ticks, size = 20)
    ax_[0].set_xticks(x_ticks)
    ax_[0].set_xticklabels(x_ticks, size=20)
    ax_[0].legend(fontsize = 20, loc=3)
    for a,b in zip(x_ticks, ax):
        ax_[0].text(a, b+0.05, '{:.2f}'.format(b), ha='center', va='bottom', fontsize=10)    
    
    ax_[1].bar(x_ticks, ay, width, label='CNN', color='teal')
    ax_[1].set_ylabel('Accelerate R$^{2}$ Y', size=25)
    ax_[1].set_yticks(y_ticks)
    ax_[1].set_yticklabels(y_ticks, size = 20)
    ax_[1].set_xticks(x_ticks)
    ax_[1].set_xticklabels(x_ticks, size=20)
    ax_[1].legend(fontsize = 20, loc=3)
    for a,b in zip(x_ticks, ay):
        ax_[1].text(a, b+0.05, '{:.2f}'.format(b), ha='center', va='bottom', fontsize=10)    
    
    plt.tight_layout()
    plt.savefig('Compare_acc.png')