import pickle
import numpy as np
from aux_fun import remove_SPH , construct_target_B, performance
from regularization import get_firing_power, alarm_generation, alarm_processing
from evaluation import alarm_evaluation, sensitivity, FPR_h, statistical_validation
from datetime import timedelta as dt
import time as t
import os

def main_testB(patient, approach, data_list, times_list, info, SPH, window_size):
    
    print('\nTesting...')
    
    start_time = t.time()
    
    # --- Results path directory ---
    path_patient = f'Results/{approach}/Patient {patient}'
    isExist = os.path.exists(path_patient)
    if not isExist:
      os.makedirs(path_patient)    
    
    # --- Load Models ---
    with open(f'../Models/{approach}/model_patient{patient}', 'rb') as models_file:
        models = pickle.load(models_file)
             
    # --- Configurations ---
    n_seizures = len(data_list)
    firing_power_threshold = 0.5
    n_events = 3
    info_test = {}
    test_prediction = {}
    
    # --- Remove SPH ---
    data_list = [remove_SPH(data_list[seizure], times_list[seizure], SPH, info['seizure'][seizure]) for seizure in range(n_seizures)]
    times_list = [times_list[seizure][:data_list[seizure].shape[0]] for seizure in range(n_seizures)]


    for SOP in models:
        
        test_prediction_per_SOP = {}
        alarm_per_SOP = {}
        info_test_per_SOP = []
            
        # ---- Approach B (0 - interictal | 1 - event) ---
        target_list = construct_target_B(data_list, times_list, SOP, n_events)
        
        #--- Model ---    
        models_SOP = models[SOP]
                
        for event in models_SOP:
            
            test_prediction_per_SOP_per_event = {}
            prediction_per_seizure = []
            alarm_per_seizure = []
            firing_power_per_seizure = []
            
            models_event = models_SOP[event]
            
            for seizure in range(n_seizures): 
                
                # --- Test ---       
                prediction = test(data_list[seizure], models_event)
        
                # --- Regularization [Firing Power + Alarms] ---
                firing_power = get_firing_power(prediction, times_list[seizure], SOP, window_size)
                alarm = alarm_generation(firing_power, firing_power_threshold)
                            
                firing_power_per_seizure.append(firing_power)
                prediction_per_seizure.append(prediction)
                alarm_per_seizure.append(alarm)
            
            # --- Concatenate seizures ---
            target = np.concatenate(target_list[event])
            prediction = np.concatenate(prediction_per_seizure)
            alarm = np.concatenate(alarm_per_seizure)
                        
            # --- Performance [samples] ---
            ss_samples, sp_samples, metric = performance(target, prediction)
                
            #--- Save parameters & results ---   
            info_test_per_SOP.append([event, ss_samples, sp_samples])
            alarm_per_SOP[f'alarm_{event}'] = alarm
            
            test_prediction_per_SOP_per_event['target'] = target_list[event]
            test_prediction_per_SOP_per_event['firing power'] = firing_power_per_seizure
            test_prediction_per_SOP_per_event['firing power threshold'] = firing_power_threshold
            test_prediction_per_SOP_per_event['alarm'] = alarm_per_seizure
            
            test_prediction_per_SOP[event] = test_prediction_per_SOP_per_event
        
        # --- Concatenate seizures (final target) --- 
        last_event = event
        final_target = np.concatenate(target_list[last_event])
        
        # --- Final alarm considering the events chronology ---
        final_alarm = construct_alarm_B(alarm_per_SOP, times_list, SOP)
        final_alarm, refractory_samples = alarm_processing(final_alarm, np.concatenate(times_list), SOP, SPH)

        # --- Performance [alarms] ---
        true_alarm, false_alarm = alarm_evaluation(final_target, final_alarm)
        ss = sensitivity(true_alarm, n_seizures)
        FPRh = FPR_h(final_target, false_alarm, refractory_samples, window_size)
        
        # --- Statistical validation ---
        surr_ss_mean, surr_ss_std, tt, pvalue, surr_ss_list = statistical_validation(target_list[last_event],final_alarm, ss, firing_power_threshold) 
               
        print(f'\n--- TEST PERFORMANCE [SOP={SOP}] --- \nSS = {ss:.3f} | FPR/h = {FPRh:.3f}')
        print(f'--- Statistical validation ---\nSS surr = {surr_ss_mean:.3f} ± {surr_ss_std:.3f} (p-value = {pvalue:.4f})')
        
        #--- Save parameters & results --- 
        test_prediction_per_SOP['final'] = [final_alarm]
        test_prediction[SOP] = test_prediction_per_SOP
        
        info_test_per_SOP.append([true_alarm, false_alarm, ss, FPRh, surr_ss_mean, surr_ss_std, tt, pvalue])
        info_test[SOP] = info_test_per_SOP
    
        # --- Save prediction values ---   
        fw = open(path_patient+f'/test_prediction_patient{patient}', 'wb')
        pickle.dump(test_prediction,fw)
        fw.close()   
    
    
    end_time = t.time()
    run_time = end_time - start_time
    print(f'\nrunning time: {run_time:.2f}')
    print('\nTested\n\n')


    return info_test




def test(data_list,models_SOP):
    
    n_classifiers = len(models_SOP['svm'])
    predictions = []
    
    for classifier in range(n_classifiers): #prediction for the 31 trained classifiers for each SOP

        # --- Model ---
        scaler = models_SOP['scaler'][classifier]
        selector = models_SOP['selector'][classifier]
        svm_model = models_SOP['svm'][classifier]
        
        # --- Standardization ---
        set_test=scaler.transform(data_list)
        
        # --- Feature Selection ---
        set_test=selector.transform(set_test)
    
        # --- Test ---
        prediction = svm_model.predict(set_test)
        predictions.append(prediction)
        
    predictions = np.array(predictions)
    
    # --- Voting System ---
    final_prediction = []
    for sample_i in range(predictions.shape[1]):
        prediction_sample_i=np.bincount(predictions[:,sample_i]).argmax() 
        final_prediction.append(prediction_sample_i)
        
    final_prediction = np.array(final_prediction)

    return final_prediction



def construct_alarm_B (alarms, times_list, SOP):
    
    datetimes=np.concatenate(times_list)
    
    idx_final_alarm=[]
    final_alarm = np.zeros(len(datetimes), dtype=int)
    
    # --- Sort alarms by event ---
    alarm_keys = list(alarms.keys())
    alarm_keys.sort()
    
    alarm_ev1 = alarms[alarm_keys[0]] #alarm event 1

    i = 0
    while i<len(alarm_ev1):
        
        if alarm_ev1[i] == 1:
            time_i = datetimes[i]
            time_i_1 = datetimes[i]
            
            event_j = 1
            is_empty=False
            
            while (event_j<len(alarm_keys)) & (is_empty==False):
                
                # --- Searching window ---
                if (event_j==1):
                    time_f = time_i + dt(minutes=2*SOP)
                else:
                    time_f = time_i_1 + dt(minutes=3*SOP)
                idx_search=np.where((datetimes>time_i) & (datetimes<=time_f))[0]
                
                if len(idx_search)==0:
                    is_empty=True
                    i+=1
                else:
                    # --- Alarm event j (2 and 3) in the searching window ---
                    alarm_evj=alarms[alarm_keys[event_j]][idx_search]
                    alarm_1=(alarm_evj==1)
                    idx_alarm_evj=idx_search[alarm_1]
    
                    if len(idx_alarm_evj)==0:
                        is_empty=True
                        i=idx_search[-1]+1 
                    else:
                        time_i=datetimes[idx_alarm_evj[0]]
                        event_j+=1
                    
                        if event_j==(len(alarm_keys)):
                            idx_final_alarm.append(idx_alarm_evj[0]) #an alarm is raised when the three events are chronologically detected
                            i=idx_alarm_evj[0]+1
        else:
            i+=1
              
    final_alarm[idx_final_alarm]=1
    
    return final_alarm

    
            
