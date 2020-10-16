#%% Import modules
import h5py
import numpy as np
from os import listdir

def get_data(dpath, data_list):  
    #%% Parameters
    bin_width = 0.064
    div = int(bin_width / 0.004)
    order = 20
    
    SESSION_UNSORTING_FR_64 = []
    FR_ORDER = []
    
    FR_ORDER_TRAIN = []
    POS_NOR_ORDER_TRAIN = []
    VEL_NOR_ORDER_TRAIN = []
    ACC_NOR_ORDER_TRAIN = []
    
    FR_ORDER_TEST = []
    POS_TEST = []
    VEL_TEST = []
    ACC_TEST = []
    
    POS_MEAN = []
    POS_MAX = []
    POS_MIN = []
    VEL_MEAN = []
    VEL_MAX = []
    VEL_MIN = []
    ACC_MEAN = []
    ACC_MAX = []
    ACC_MIN = []
    
    TOTL_POS_X = []
    TOTL_POS_Y = []
    TOTL_VEL_X = []
    TOTL_VEL_Y = []
    TOTL_ACC_X = []
    TOTL_ACC_Y = []
    
    TBIN = []
    
    
    #%% Channel based
    ch_list = []
    for k in range(96):
        ch_list.append(k+1)
    
    print('-----------------')
                
    #%% Functions
    # Get firing rate
    def _getfrate(x):
        if x.shape[0] != 2:
            fr, _ = np.histogram(x[0], tbin)
        else:    # SPIKES is empty
            fr = np.zeros([1,tbin.shape[0]-1])
        
        return fr
    
    # Normalize
    def _norMove(x, train = True, x_mean = None, x_max = None, x_min = None): 
        # x dim.
        if train == True:
            x_nor_x = np.zeros([len(x[:,0]), 1])
            x_mean_x = np.mean(x[:,0])
            x_max_x = np.max(x[:,0])
            x_min_x = np.min(x[:,0])
            
            x_nor_x = (x[:,0] - x_mean_x) / (x_max_x - x_min_x)       
        else:
            x_nor_x = (x[:,0] - x_mean_x) / (x_max_x - x_min_x)
        # y dim.   
        if train == True:
            x_nor_y = np.zeros([len(x[:,1]), 1])
            x_mean_y = np.mean(x[:,1])
            x_max_y = np.max(x[:,1])
            x_min_y = np.min(x[:,1])
            
            x_nor_y = (x[:,1] - x_mean_y) / (x_max_y - x_min_y)       
        else:
            x_nor_y = (x[:,1] - x_mean_y) / (x_max_y - x_min_y)
        # concatenate
        x_nor = np.vstack((x_nor_x, x_nor_y)).T
        x_mean = np.vstack((x_mean_x, x_mean_y)).T
        x_max = np.vstack((x_max_x, x_max_y)).T
        x_min = np.vstack((x_min_x, x_min_y)).T
        
        return x_nor, x_mean, x_max, x_min
    
    # Order
    def _2order(x_time_chn, order):
        pre_data = np.zeros([x_time_chn.shape[0], order, x_time_chn.shape[1]])
        pre_order = np.zeros([order-1, x_time_chn.shape[1]])
        data = np.concatenate((pre_order, x_time_chn), axis = 0)
        for kkk in range(order):
            pre_data[:, kkk, :] = data[kkk:x_time_chn.shape[0] + kkk, :]
        
        return pre_data
    
    #%% Get data
    for s in range(len(data_list)):
        # Read data
        fname = data_list[s]
        file = dpath + '/' + fname
        
        mat_file = h5py.File(file,'r')
        CHANNELS = mat_file[list(mat_file.keys())[1]]
        FINGER_POS = mat_file[list(mat_file.keys())[3]]
        TIMES = mat_file[list(mat_file.keys())[5]]
        SPIKES = mat_file[list(mat_file.keys())[4]]
        
        NumUni = SPIKES.shape[0]
        
        tbin = np.array(TIMES[0][::div])
          
        #%% Kinematic var.
        pos_x = np.array(FINGER_POS[1][::div]) * -10
        pos_y = np.array(FINGER_POS[2][::div]) * -10
        pos = np.vstack((pos_x, pos_y)).T
        
        vel = (pos[1:] - pos[:-1]) / bin_width
        
        acc = (vel[1:] - vel[:-1]) / bin_width
        acc = np.concatenate((np.zeros([1,2]), acc), axis = 0)
          
        #%% Firing rate  
        # Initail parameters
        frate_hash = np.zeros([1,tbin.shape[0]-1])
        frate_unit_01 = np.zeros([1,tbin.shape[0]-1])
        frate_unit_02 = np.zeros([1,tbin.shape[0]-1])
        frate_unit_03 = np.zeros([1,tbin.shape[0]-1])
        frate_unit_04 = np.zeros([1,tbin.shape[0]-1])
        # Compute frate along with channel
        for ch_index in ch_list:
            unit_hash = mat_file[SPIKES[0][ch_index-1]] 
            unit_01 = mat_file[SPIKES[1][ch_index-1]]
            unit_02 = mat_file[SPIKES[2][ch_index-1]]
            
            fr_unit_hash = _getfrate(unit_hash)
            fr_unit_01 = _getfrate(unit_01)
            fr_unit_02 = _getfrate(unit_02)
    
            if NumUni == 5:
                unit_03 = mat_file[SPIKES[3][ch_index-1]]
                unit_04 = mat_file[SPIKES[4][ch_index-1]]
                
                fr_unit_03 = _getfrate(unit_03)
                fr_unit_04 = _getfrate(unit_04)
            
            else:
                fr_unit_03 = np.zeros([1,tbin.shape[0]-1])
                fr_unit_04 = np.zeros([1,tbin.shape[0]-1])
            
            frate_hash = np.vstack((frate_hash, fr_unit_hash))
            frate_unit_01 = np.vstack((frate_unit_01, fr_unit_01))
            frate_unit_02 = np.vstack((frate_unit_02, fr_unit_02))
            frate_unit_03 = np.vstack((frate_unit_03, fr_unit_03))
            frate_unit_04 = np.vstack((frate_unit_04, fr_unit_04))
                
        # Delete first row
        frate_hash = np.delete(frate_hash, 0, 0)
        frate_unit_01 = np.delete(frate_unit_01, (0), axis = 0)
        frate_unit_02 = np.delete(frate_unit_02, (0), axis = 0)
        frate_unit_03 = np.delete(frate_unit_03, (0), axis = 0)
        frate_unit_04 = np.delete(frate_unit_04, (0), axis = 0)
        
        # Concatenate
        frate_unsort = frate_hash + frate_unit_01 + frate_unit_02 + frate_unit_03 + frate_unit_04
        frate_sort =  np.concatenate((frate_hash.reshape(1, frate_hash.shape[0], frate_hash.shape[1]),
                                    frate_unit_01.reshape(1, frate_unit_01.shape[0], frate_unit_01.shape[1]),
                                    frate_unit_02.reshape(1, frate_unit_02.shape[0], frate_unit_02.shape[1]),
                                    frate_unit_03.reshape(1, frate_unit_03.shape[0], frate_unit_03.shape[1]),
                                    frate_unit_04.reshape(1, frate_unit_04.shape[0], frate_unit_04.shape[1])),axis = 0)   
        
        #%% Cut data
        # Choose sorting or unsorted data to use
        frate = np.copy(frate_unsort).transpose()    # time by chn
        use_length = frate.shape[0]
        train_length = int(320 / bin_width)
        
        tbin_cut = tbin.reshape([len(tbin),1])[1:use_length+1,:]    
        
        pos = pos[1:use_length+1, :]
        
        
        #%% Normalization
        pos_nor, pos_mean, pos_max, pos_min = _norMove(pos[0:train_length, :], train=True)
        vel_nor, vel_mean, vel_max, vel_min = _norMove(vel[0:train_length, :], train=True)
        acc_nor, acc_mean, acc_max, acc_min = _norMove(acc[0:train_length, :], train=True)
        
        #%% Divide data   
        # training set
        frate_train= frate[0:train_length, :]
        pos_nor_train = pos_nor[0:train_length, :]
        vel_nor_train = vel_nor[0:train_length, :] 
        acc_nor_train = acc_nor[0:train_length, :] 
        
        # testing set
        frate_test = frate[train_length:, :]
        pos_test = pos[train_length:, :]
        vel_test = vel[train_length:, :] 
        acc_test = acc[train_length:, :]
        
        #%% Order
        frate_train_order = _2order(frate_train, order)
        pos_train_order = _2order(pos_nor_train, order)
        vel_train_order = _2order(vel_nor_train, order)
        acc_train_order = _2order(acc_nor_train, order)
        
        frate_test_order = _2order(frate_test, order)
        
        #%% All FR
        tot = np.concatenate((frate_train_order, frate_test_order), axis=0)
        
        #%% Save var.
        FR_ORDER_TRAIN.append(frate_train_order)
        POS_NOR_ORDER_TRAIN.append(pos_train_order)
        VEL_NOR_ORDER_TRAIN.append(vel_train_order)
        ACC_NOR_ORDER_TRAIN.append(acc_train_order)
            
        FR_ORDER_TEST.append(frate_test_order)
        POS_TEST.append(pos_test)
        VEL_TEST.append(vel_test)
        ACC_TEST.append(acc_test)
            
        POS_MEAN.append(pos_mean)
        POS_MAX.append(pos_max)
        POS_MIN.append(pos_min)
        VEL_MEAN.append(vel_mean)
        VEL_MAX.append(vel_max)
        VEL_MIN.append(vel_min)
        ACC_MEAN.append(acc_mean)
        ACC_MAX.append(acc_max)
        ACC_MIN.append(acc_min)
            
        TBIN.append(tbin_cut)
        
        TOTL_POS_X.append(pos[:,0])
        TOTL_POS_Y.append(pos[:,1])
        TOTL_VEL_X.append(vel[:,0])
        TOTL_VEL_Y.append(vel[:,1])
        TOTL_ACC_X.append(acc[:,0])
        TOTL_ACC_Y.append(acc[:,1])
        
        # Else
        SESSION_UNSORTING_FR_64.append(frate)
        FR_ORDER.append(tot)
        
        #%% Print result    
        print('\n------Info------')
        print('file_name: '+ fname)
        print('Channel: ', CHANNELS.shape[1])
        print('Units: ', NumUni)
        print('Unsorted: ', frate_unsort.shape)
        print('Sorting: ', frate_sort.shape)
        print('Org_Time: ', TIMES.shape[1])
        print('Bin_width: ', bin_width)
        print('Tbin: ', len(tbin))
        print('Tbin_start: ' + str(tbin[0]))
        print('Tbin_end: ' + str(tbin[-1]))
        print('Order: ', order)
        print('Training: ', POS_NOR_ORDER_TRAIN[s].shape[0])
        print('Testing: ', POS_TEST[s].shape[0])
        print('------------------')

    # Dict
    A = {
      "FR_ORDER_TRAIN": FR_ORDER_TRAIN,
      "POS_NOR_ORDER_TRAIN": POS_NOR_ORDER_TRAIN,
      "VEL_NOR_ORDER_TRAIN": VEL_NOR_ORDER_TRAIN,
      "ACC_NOR_ORDER_TRAIN": ACC_NOR_ORDER_TRAIN,
      "FR_ORDER_TEST": FR_ORDER_TEST,
      "POS_TEST": POS_TEST,
      "VEL_TEST": VEL_TEST,
      "ACC_TEST": ACC_TEST,
    }
    B = {
      "POS_MAX": POS_MAX,
      "POS_MIN": POS_MIN,
      "POS_MEAN": POS_MEAN,
      "VEL_MAX": VEL_MAX,
      "VEL_MIN": VEL_MIN,
      "VEL_MEAN": VEL_MEAN,
      "ACC_MAX": ACC_MAX,
      "ACC_MIN": ACC_MIN,
      "ACC_MEAN": ACC_MEAN,
    }
  
    return A, B