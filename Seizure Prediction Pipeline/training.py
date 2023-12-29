# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 19:01:25 2021

@author: User
"""
import numpy as np
import pickle
import time as t
from aux_fun import remove_SPH, construct_target_A, construct_target_B, construct_target_C, class_balancing, standardization, select_features, classifier, performance

def main_train(patient, approach, data_list, times_list, info, SPH, SOP_list, window_size):
    
    # --- Configurations ---
    n_events = 3
    n_seizures = len(data_list)
    models = {}
    info_train = []
    
    for SOP in SOP_list:
        
        print(f'\n------------ SOP={SOP}min ------------')
        
        models_SOP = {}
        
        # --- Remove SPH ---
        data_list = [remove_SPH(data_list[seizure], times_list[seizure], SPH, info['seizure'][seizure]) for seizure in range(n_seizures)]

        # --- Construct target ---
        target_list = {}
        
        if approach == 'Approach A':
            # ---- Approach A (0 - interictal | 1 - preictal) ---
            target = [construct_target_A(data_list[seizure], times_list[seizure], SOP) for seizure in range(n_seizures)]
            target_list['Standard'] = target
            
        elif approach=='Approach B':
            # ---- Approach B (0 - interictal | 1 - preictal event) ---
            target_list = construct_target_B(data_list, times_list, SOP, n_events)
            
        elif approach=='Approach C':
            # --- Approach C (0 - interictal | 1 - preictal event) ---
            target_list = construct_target_C(data_list, times_list, SOP, n_events)        
        
        
        # GRID SEARCH | TRAIN | SAVE MODELS
        for event in target_list:
                        
            if approach!='Approach A':
                print(f'\n--------- {event} ---------')
                
            # --- Grid search ---
            best_k, best_C, ss, sp, best_metric = grid_search(data_list, target_list[event])
            
            # --- Train ---
            model = train(data_list, target_list[event], best_k, best_C)
            
            # --- Save parameters & results ---
            if approach=='Approach A':
                models_SOP=model
                info_train.append([SOP, best_k, best_C, ss, sp, best_metric])
            else:
                models_SOP[event] = model
                info_train.append([SOP, event, best_k, best_C, ss, sp, best_metric])
        
    # --- Save models ---
        models[SOP]=models_SOP
    fw=open(f'Models/{approach}/model_patient{patient}', 'wb')
    pickle.dump(models,fw)
    fw.close()
    
    return info_train
    


def grid_search(data, target):

    print('\nGrid search...')
    
    # --- Configurations ---
    k = np.arange(start = 10, stop = 41, step = 10) # number of features
    c_pot = np.arange(start = -10, stop = 11, step = 2, dtype=float)
    C = 2**c_pot # SVM cost parameter
    n_classifiers = 31 # number of classifiers (odd to avoid ties)
    
    # --- Cross validation ----
    n_folds = len(data) # number of folds
   
    performances = []
    for k_i in k:
        for C_i in C:
            print(f'\n--------- k = {k_i} | C = {C_i:.2g} ---------')
            
            start_time_combination = t.time()
    
            ss_per_combination = []
            sp_per_combination = []
            metric_per_combination = [] 
            
            for fold_i in range(n_folds): 
                for classifier_i in range(n_classifiers):
            
                    # --- Data splitting (train & validation) ---
                    set_validation = data[fold_i]
                    set_train = [data[fold_j] for fold_j in range(n_folds) if fold_j!=fold_i]
                    target_validation = target[fold_i]
                    target_train = [target[fold_j] for fold_j in range(n_folds) if fold_j!=fold_i]
                    
                    # --- Class balancing (per seizure) ---
                    idx_selected = [class_balancing(target_train[seizure]) for seizure in range(len(target_train))]
                    set_train = np.concatenate([set_train[seizure][idx_selected[seizure]] for seizure in range(len(idx_selected))])
                    target_train = np.concatenate([target_train[seizure][idx_selected[seizure]] for seizure in range(len(idx_selected))])
                    
                
                    # --- Standardization ---
                    scaler = standardization(set_train)
                    set_train = scaler.transform(set_train)
                    set_validation = scaler.transform(set_validation)

                    # --- Feature selection ---
                    selector = select_features(set_train, target_train, k_i)
                    set_train = selector.transform(set_train)
                    set_validation = selector.transform(set_validation)

                    # --- Train (training set) ---
                    svm_model =  classifier(set_train, target_train, C_i)
                    
                    # --- Test (validation set) ---
                    prediction_validation = svm_model.predict(set_validation)
                     
                    # --- Performance ---
                    ss, sp, metric = performance(target_validation, prediction_validation)
                    
                    ss_per_combination.append(ss) 
                    sp_per_combination.append(sp)
                    metric_per_combination.append(metric)
                    
            ss_avg = np.mean(ss_per_combination)
            sp_avg = np.mean(sp_per_combination)
            metric_avg = np.mean(metric_per_combination)
            print(f'Average performance: SS = {ss_avg:.2f} | SP = {sp_avg:.2f} | metric: {metric_avg:.2f}')
    
            end_time_combination = t.time()
            run_time = end_time_combination - start_time_combination
            print(f'Running time per combination = {run_time:.2f}')
            
            performances.append([k_i, C_i, ss_avg, sp_avg, metric_avg, run_time])

    performances = np.array(performances)
        
    
    # --- Select best parameters ---
    
    # Best performance (maximum metric)
    best_performance = max(performances[:,4])
    idx_best = np.where(performances[:,4] == best_performance)[0]
    
    # Tiebreaker (minimum running time)          
    if len(idx_best)>1:
        tiebreaker_performance = min(performances[idx_best,5])
        idx_best_tie = np.where(performances[idx_best,5] == tiebreaker_performance)[0][0]
        idx_best = idx_best[idx_best_tie]
    else:
        idx_best = idx_best[0]
    
    # --- Save selected parameters & results ---
    best_k = performances[idx_best,0]
    best_C = performances[idx_best,1]
    ss = performances[idx_best,2]
    sp = performances[idx_best,3]
    best_metric = performances[idx_best,4]  # best_performance
    
    print('\nGrid search completed')
    print(f'\n --------------- GRID SEARCH (best result) ---------------\nk = {best_k:.2f} | C = {best_C:.2f} | SS = {ss:.2f} | SP = {sp:.2f} | metric = {best_metric:.2f}')
        
    return best_k, best_C, ss, sp, best_metric




def train(data, target, best_k, best_C):
    
    print('\n\nTraining classifier...')
    
    start_time = t.time()
    n_classifiers = 31
    
    scaler_list = []
    selector_list = []
    svm_list = []
    
    
    for classifier_i in range(n_classifiers):
    
        # --- Class balancing (per seizure) ---
        idx_selected = [class_balancing(target[seizure]) for seizure in range(len(target))]
        data_train = np.concatenate([data[seizure][idx_selected[seizure]] for seizure in range(len(idx_selected))])
        target_train = np.concatenate([target[seizure][idx_selected[seizure]] for seizure in range(len(idx_selected))])
                
        # --- Standardization ---
        scaler = standardization(data_train)
        data_train = scaler.transform(data_train)

        # --- Feature selection ---
        selector = select_features(data_train, target_train, int(best_k))      
        data_train = selector.transform(data_train)

        # --- Train (training set) ---
        svm_model = classifier(data_train, target_train, best_C)
        
        # --- Save --- 
        scaler_list.append(scaler)
        selector_list.append(selector)
        svm_list.append(svm_model)
                    
    # --- Model ---
    model = {}
    model['scaler'] = scaler_list
    model['selector'] = selector_list
    model['svm'] = svm_list


    end_time = t.time()
    run_time = end_time - start_time
    print(f'running time: {run_time:.2f}')
    print('Classifier trained')

    return model