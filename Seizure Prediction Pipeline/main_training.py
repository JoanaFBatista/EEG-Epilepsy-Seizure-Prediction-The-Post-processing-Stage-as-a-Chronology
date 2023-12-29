# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:03:56 2021

@author: User
"""

from import_data import import_data
from aux_fun import splitting
from training import main_train
from save_results import save_train_results_B_C, save_train_results_A

# %% --- Configurations ---

# Approach A --> controlo
# Approch B --> chronological firing power
# Approch C --> cumulative firing power

approach_list = ['Approach A', 'Approach B', 'Approach C']

for approach in approach_list:
    
    patient_list=['402','8902','11002','16202','21902','23902','26102',
    '30802','32702','45402','46702','50802','52302','53402','55202','56402',
    '58602','59102','60002','64702','75202','80702','85202','93402','93902',
    '94402','95202','96002','98102','98202','101702','102202','104602',
    '109502','110602','112802','113902','114702','114902','123902']
     
    # --- Train options ---
    n_seizures_train = 3
    SPH = 5 
    SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    window_size=5 
    
    general_information = [n_seizures_train, SPH]
    
    # %% Import Data | Siplitting Data | Train 
    
    train_information={}
    
    print(f'\n----------------- {approach} -----------------')
    
    for patient in patient_list:
        
        print(f'\n------------- Patient:{patient} -------------')
    
        # --- Import data ---
        features, datetimes, features_channels, feature_names, channel_names, seizure_info = import_data(patient)
        
        # --- Splitting data (train|test) ---
        data,datetimes,seizure_info = splitting(features, datetimes, seizure_info, n_seizures_train)
        
        # --- Train ---
        info_tr={};
        info_tr['seizure'] = seizure_info['train']
        info_tr['features'] = features_channels
        
        info_train_patient = main_train(patient, approach, data['train'], datetimes['train'], info_tr, SPH, SOP_list, window_size)
        
        train_information[patient] = info_train_patient
    
    # --- Save results (excel) ---
    if approach == 'Approach A':       
        save_train_results_A(general_information, train_information, approach)
    else:
        save_train_results_B_C(general_information, train_information, approach)
    
    

