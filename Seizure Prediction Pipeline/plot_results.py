import matplotlib.pyplot as plt 
import numpy as np
import pickle
from aux_fun import get_selected_features
import pandas as pd
import seaborn as sns


# --- Bar plot with SS and FPR/h for each patient considering all SOPs--- 
def fig_performance(patient, SOP_list , info_test, approach):
    # Information
    ss=[]
    fprh=[]
    
    for SOP in info_test:
        if approach=='Approach A':
            ss.append(info_test[SOP][4])
            fprh.append(info_test[SOP][5])
        else:
            ss.append(info_test[SOP][-1][2])
            fprh.append(info_test[SOP][-1][3])
        
    labels=[str(SOP) for SOP in info_test]  

    # Figure
    fig = plt.figure(figsize=(20, 15))
    
    x = np.arange(len(ss))
    width = 0.3
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='#E3797D', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.xlabel('SOP')
    plt.ylabel('SS', color='#E3797D', fontweight='bold')
    plt.ylim([0, 1.05])
    
    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='#82A69C', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='#82A69C', fontweight='bold')
    
    plt.xticks(x, labels=labels)
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)

    plt.title(f'Performance (patient {patient})', fontsize=20)

    plt.savefig(f'Results/{approach}/Patient {patient}/Performance (patient {patient}, all SOPs)', bbox_inches='tight')
    plt.close()    
    

# --- Bar plot with SS and FPR/h for each SOP considering all patients --- 
def fig_performance_all_patient(SOP_list , info_test, approach):
    
    # Information
    for SOP in SOP_list:
        ss=[]
        fprh=[]
        for patient in info_test:
            if approach=='Approach A':
                ss.append(info_test[patient][SOP][4])
                fprh.append(info_test[patient][SOP][5])
            else:
                ss.append(info_test[patient][SOP][-1][2])
                fprh.append(info_test[patient][SOP][-1][3])
        
        labels=[str(patient) for patient in info_test]  

        # Figure
        fig = plt.figure(figsize=(20, 15))
        
        x = np.arange(len(ss))
        width = 0.3
        
        # Sensitivity
        ss_bar = plt.bar(x-width/2, ss, width, color='#E3797D', label = 'SS')
        plt.bar_label(ss_bar, fmt='%.2f', padding=3)
        plt.xlabel('SOP')
        plt.ylabel('SS', color='#E3797D', fontweight='bold')
        plt.ylim([0, 1.05])
        
        # FPR/h
        plt.twinx() # second axis
        fprh_bar = plt.bar(x+width/2, fprh, width, color='#82A69C', label = 'FPR/h')
        plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
        plt.ylabel('FPR/h', color='#82A69C', fontweight='bold')
        
        plt.xticks(x, labels=labels, rotation = 90)
        fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    
        plt.title(f'Performance (All patients - SOP={SOP})', fontsize=20)
    
        plt.savefig(f'Results/{approach}/Figures_allPatients/Performance (All patients, SOP={SOP})', bbox_inches='tight')
        plt.close()  


# --- Selection frequency of each feature for approach A ---
def fig_feature_selection_A(patients, approach, features, channels, features_channels):
    
    idx_selected_features_all_patients = {}
    
    for patient in patients:
        
        idx_selected_features_list = {}
        
        #Load models
        models = pickle.load(open(f'Models/{approach}/model_patient{patient}','rb'))
        
        for SOP in models:
            # concatenate features from 31 models
            idx_selected_features = np.concatenate([get_selected_features(len(features_channels),models[SOP]['selector'][classifier]) for classifier in range(len(models[SOP]['selector']))]) 
            idx_selected_features_list[SOP] = idx_selected_features
    
            # Figure: selected features for each patient and SOP
            fig_fs_A(features, channels, features_channels, idx_selected_features, f'patient{patient}_SOP{SOP}',patient, approach, SOP)
        
        idx_selected_features_all_patients[patient] = idx_selected_features_list
    
    for SOP in models:     
        idx_features = np.concatenate([idx_selected_features_all_patients[patient][SOP] for patient in patients])
        
        # Figure: selected features for each SOP considering all patients
        fig_fs_A(features, channels, features_channels, idx_features, f'all_patients_SOP{SOP}', None, approach, SOP)
        
        
        
    
def fig_fs_A(features, channels, features_channels, idx_selected_features, fig_name, patient, approach, SOP):
    if patient is not None:
        path_to_save=f'Results/{approach}/Patient {patient}'
        title=f'Pratient: {patient} - SOP: {SOP} min'
    else:
        path_to_save=f'Results/{approach}/Figures_allPatients'
        title=f'All patients - SOP: {SOP} min'
    
    # selection frequency for each feature and channel
    df = pd.DataFrame(data = 0, index = channels, columns = features)
    for i in idx_selected_features:
        idx_feature = i//len(channels)
        idx_channel = i%len(channels)        
        df.iloc[idx_channel, idx_feature]+=1
    df=df.div(len(idx_selected_features)) # translate from number of occurrences to relative frequency
    
    plt.figure(figsize=(17, 10))
    plt.title(f'Relative frequency of selected features and channels\n{title}', fontsize=13)
    sns.heatmap(df, cmap=sns.light_palette('#6d8a82', as_cmap=True),  vmin=0, vmax=0.2, linecolor='white',linewidths=0.5, xticklabels=True, yticklabels=True)

    plt.subplots_adjust(top=0.91,
        bottom=0.23,
        left=0.05,
        right=1.0,
        hspace=0.2,
        wspace=0.2)

    plt.savefig(path_to_save+f'/feature_selection_{fig_name}')
    plt.close()


# --- Selection frequency of each feature for approach B and C ---
def fig_feature_selection_B_C(patients, approach, features, channels, features_channels):
    
    idx_selected_features_all_patients = {}
    
    for patient in patients:
        
        idx_selected_features_list = {}
        
        #Load models
        models = pickle.load(open(f'Models/{approach}/model_patient{patient}','rb'))
        
        for SOP in models:
            
            idx_selected_features_event = {}
            
            for event in models[SOP]:
                # concatenate features from 31 models
                
                idx_selected_features_event[event] = np.concatenate([get_selected_features(len(features_channels),models[SOP][event]['selector'][classifier]) for classifier in range(len(models[SOP][event]['selector']))]) 
            idx_selected_features_list[SOP] = idx_selected_features_event

            # Figure: selected features for each patient and SOP
            fig_fs_B_C(features, channels, features_channels, idx_selected_features_event, f'patient{patient}_SOP{SOP}',patient, approach, SOP)
            
        idx_selected_features_all_patients[patient] = idx_selected_features_list
    
    for SOP in models:
        idx_selected_features = {}
        for event in models[SOP]:
            idx_selected_features[event] = np.concatenate([idx_selected_features_all_patients[patient][SOP][event] for patient in patients])
        
        # Figure: selected features for each SOP considering all patients
        fig_fs_B_C(features, channels, features_channels, idx_selected_features, f'all_patients_SOP{SOP}', None, approach, SOP)
        
    
def fig_fs_B_C(features, channels, features_channels, idx_selected_features, fig_name, patient, approach, SOP):
    
    if patient is not None:
        path_to_save=f'Results/{approach}/Patient {patient}'
        title=f'Pratient: {patient} - SOP: {SOP} min'
    else:
        path_to_save=f'Results/{approach}/Figures_allPatients'
        title=f'All patients - SOP: {SOP} min'
    
    fig, axs = plt.subplots(3, sharex=True, figsize=(23, 10))
    fig.suptitle(f'Relative frequency of selected features and channels\n{title}', fontsize=13)
    cbar_ax = fig.add_axes([.95, .18, .01, .75])
    
    j=0
    for event in idx_selected_features:
        
        # selection frequency for each feature and channel
        df = pd.DataFrame(data = 0, index = channels, columns = features)
        for i in idx_selected_features[event]:
            idx_feature = i//len(channels)
            idx_channel = i%len(channels)        
            df.iloc[idx_channel, idx_feature]+=1
     
        df = df.div(len(idx_selected_features[event])) # translate from number of occurrences to relative frequency
   
        if j!=2:
            sns.heatmap(df, cmap=sns.light_palette('#6d8a82', as_cmap=True), cbar=None,  vmin=0, vmax=0.2, linecolor='white',linewidths=0.5, xticklabels=True, yticklabels=True, ax=axs[j])
            axs[j].set(xlabel="", ylabel=event)
        else:
            sns.heatmap(df, cmap=sns.light_palette('#6d8a82', as_cmap=True), cbar_ax=cbar_ax, vmin=0, vmax=0.2, linecolor='white',linewidths=0.5, xticklabels=True, yticklabels=True, ax=axs[j])
            axs[j].set(xlabel="", ylabel=event)
            
        j+=1

    axs[-1].tick_params(axis = 'x', rotation=90, labelsize=8)
    fig.subplots_adjust(
        top=0.93,
        bottom=0.18,
        left=0.04,
        right=0.94,
        hspace=0.05,
        wspace=0.0)
    
    plt.savefig(path_to_save+f'/feature_selection_{fig_name}.png')
    plt.close()

