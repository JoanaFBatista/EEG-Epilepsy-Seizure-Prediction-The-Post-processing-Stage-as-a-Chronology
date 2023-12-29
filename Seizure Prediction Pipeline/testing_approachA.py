import pickle
import numpy as np
from aux_fun import remove_SPH ,construct_target_A, performance
from regularization import get_firing_power, alarm_generation, alarm_processing
from evaluation import alarm_evaluation, sensitivity, FPR_h, statistical_validation
import os

def main_testA(patient, approach, data_list, times_list, info, SPH, window_size):
    print('\nTesting...')
    
    # --- Results path directory ---
    path_patient=f'Results/{approach}/Patient {patient}'
    isExist = os.path.exists(path_patient)
    if not isExist:
      os.makedirs(path_patient)    
    
    # --- Load Models ---
    with open(f'../Models/{approach}/model_patient{patient}', 'rb') as models_file:
        models = pickle.load(models_file)
             
    # --- Configurations ---
    n_seizures = len(data_list)
    firing_power_threshold = 0.5
    info_test = {}
    test_prediction = {}
    
    # --- Remove SPH ---
    data_list = [remove_SPH(data_list[seizure], times_list[seizure], SPH, info['seizure'][seizure]) for seizure in range(n_seizures)]
    times_list = [times_list[seizure][:data_list[seizure].shape[0]] for seizure in range(n_seizures)]

    for SOP in models:
        
        test_prediction_per_SOP = {}
        prediction_per_seizure = []
        alarm_per_seizure = []
        refractory_samples_per_seizure = []
        firing_power_per_seizure = []
    
        # --- Construct target (0 - interictal | 1 - preictal) ---
        target_list = [construct_target_A(data_list[seizure], times_list[seizure], SOP) for seizure in range(n_seizures)]
        
        #--- Model ---    
        models_SOP = models[SOP]

        for seizure in range(n_seizures): 
            # --- Test ---       
            prediction = test(data_list[seizure], models_SOP)
    
            # --- Regularization [Firing Power + Alarms] ---
            firing_power = get_firing_power(prediction, times_list[seizure], SOP, window_size)
            alarm = alarm_generation(firing_power, firing_power_threshold)
            alarm, refractory_samples = alarm_processing(alarm, times_list[seizure], SOP, SPH)
                        
            firing_power_per_seizure.append(firing_power)
            prediction_per_seizure.append(prediction)
            alarm_per_seizure.append(alarm)
            refractory_samples_per_seizure.append(refractory_samples)
                                   
        # --- Concatenate seizures ---
        target = np.concatenate(target_list)
        prediction = np.concatenate(prediction_per_seizure)
        alarm = np.concatenate(alarm_per_seizure)
        refractory_samples = np.concatenate(refractory_samples_per_seizure)
                
        # --- Performance [samples] ---
        ss_samples, sp_samples, metric = performance(target, prediction)
        
        # --- Performance [alarms] ---
        true_alarm, false_alarm = alarm_evaluation(target, alarm)
        ss = sensitivity(true_alarm, n_seizures)
        FPRh = FPR_h(target, false_alarm, refractory_samples, window_size)
        
        # --- Statistical validation ---
        surr_ss_mean, surr_ss_std, tt, pvalue, surr_ss_list = statistical_validation(target_list, alarm, ss, firing_power_threshold)
        
        # --- Save parameters & results ---      
        test_prediction_per_SOP['patient'] = patient
        test_prediction_per_SOP['approach'] = approach
        test_prediction_per_SOP['target'] = target_list
        test_prediction_per_SOP['prediction'] = prediction_per_seizure
        test_prediction_per_SOP['firing power'] = firing_power_per_seizure
        test_prediction_per_SOP['firing power threshold'] = firing_power_threshold
        test_prediction_per_SOP['alarm'] = alarm_per_seizure
        
        test_prediction[SOP] = test_prediction_per_SOP
        
        info_test[SOP] = [ss_samples, sp_samples, true_alarm, false_alarm, ss, FPRh, surr_ss_mean, surr_ss_std, tt, pvalue]
        
        print(f'\n--- TEST PERFORMANCE [SOP={SOP}] --- \nSS = {ss:.3f} | FPR/h = {FPRh:.3f}')
        print(f'--- Statistical validation ---\nSS surr = {surr_ss_mean:.3f} ± {surr_ss_std:.3f} (p-value = {pvalue:.4f})')
                
    
    # --- Save prediction values ---   
    fw = open(path_patient+f'/test_prediction_patient{patient}', 'wb')
    pickle.dump(test_prediction,fw)
    fw.close()
    
    print('\nTested\n\n')
    
    return info_test


def test(data_list,models_SOP):
    
    n_classifiers = len(models_SOP['svm'])
    predictions = []
    
    for classifier in range(n_classifiers):

        # --- Model ---
        scaler = models_SOP['scaler'][classifier]
        selector = models_SOP['selector'][classifier]
        svm_model = models_SOP['svm'][classifier]
        
        # --- Standardization ---
        set_test = scaler.transform(data_list)
        
        # --- Feature Selection ---
        set_test = selector.transform(set_test)
    
        # --- Test ---
        prediction = svm_model.predict(set_test)
        predictions.append(prediction)
        
    predictions=np.array(predictions)
    
    # --- Voting System ---
    final_prediction = []
    for sample_i in range(predictions.shape[1]):
        prediction_sample_i = np.bincount(predictions[:,sample_i]).argmax() 
        final_prediction.append(prediction_sample_i)
        
    final_prediction = np.array(final_prediction)

    return final_prediction
    
            
