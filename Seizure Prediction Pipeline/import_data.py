import os
import numpy as np
import pickle
import pandas

def import_data(patient):
    
    print('\nLoading data...')
    
    directory=f'../../Data/pat_{patient}'
    
    #import information about seizures
    with open(os.path.join(directory,'all_seizure_information.pkl'), 'rb') as seiz_info:
        seizure_info = pickle.load(seiz_info)
        
    #import channel names
    with open('../../Data/channel_names.pkl','rb') as ch:
        channel_names = pickle.load(ch)
    channel_names=[channel_names[i] for i in range(19)]
        
    #import feature names
    f = open('../../Data/univariate_feature_names.txt','r')
    feature_names=f.read().splitlines()
    
        
    #import datetimes and features    
    datetimes=[]
    features=[]
    
    for seizure_n in range (len(os.listdir(os.path.join(directory, 'datetimes')))):
    
        path_times='datetimes/feature_datetimes_{}.npy'.format(seizure_n)
        path_features='features/pat_{}_seizure_{}_features.npy'.format(patient,seizure_n)
    
        times = np.load(os.path.join(directory, path_times))
        times = np.array([pandas.to_datetime(time, unit='s') for time in times])
        datetimes.append(times)
        
        # --- Data reshape (data[n_samples, 59*19]) ---
        feature=np.load(os.path.join(directory, path_features))
        feature = np.concatenate([feature[:,f,:] for f in range(feature.shape[1])], axis=1)
        features.append(feature)
    
    # --- Feature names (features:channels) ---
    features_channels=[]
    for fn in range(len(feature_names)):
        for cn in range(len (channel_names)):
            feat=feature_names[fn]+':'+channel_names[cn]
            features_channels.append(feat)
    
    print('Data loaded')  

    return features,datetimes,features_channels,feature_names,channel_names,seizure_info








