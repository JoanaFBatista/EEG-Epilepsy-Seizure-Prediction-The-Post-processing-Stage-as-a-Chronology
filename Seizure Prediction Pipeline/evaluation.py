import numpy as np
import random
from scipy import stats


def alarm_evaluation(target, alarm):
    
    idx_preictal = np.argwhere(target==1)
    idx_alarms = np.argwhere(alarm==1)
    
    true_alarm = 0
    false_alarm = 0
    
    for idx in idx_alarms:
        if idx in idx_preictal:
            true_alarm += 1
        else:
            false_alarm += 1
            
    return true_alarm, false_alarm


def sensitivity(true_alarm, n_seizures):
    
    ss = true_alarm/n_seizures
    
    return ss


def FPR_h(target, false_alarm, refractory_samples, window_size):
    
    interictal_samples = np.count_nonzero(target==0)
    interictal_duration = interictal_samples*window_size # seconds
    
    refractory_samples = np.count_nonzero(target[refractory_samples]==0)
    refractory_duration = refractory_samples*window_size # seconds

    FPRh = false_alarm/((interictal_duration - refractory_duration)/3600)    
    
    return FPRh


def statistical_validation(target_list, alarm, ss, threshold):
    
    n_seizures = len(target_list)
    n_runs = 30 # number of repetitons
    surr_ss_list = []
    for run in range(n_runs):
        
        # --- Construct surrogates target ---
        surr_target = np.concatenate([shuffle_target(target, threshold) for target in target_list])
        
        # --- Performance [alarms] ---
        true_alarm, false_alarm = alarm_evaluation(surr_target, alarm)
        surr_ss = sensitivity(true_alarm, n_seizures)
        surr_ss_list.append(surr_ss)

    surr_ss_mean = np.mean(surr_ss_list)
    surr_ss_std = np.std(surr_ss_list)
    
    # --- One sample t-test ---
    tt, pvalue = t_test(surr_ss_list, ss)
        
    return surr_ss_mean, surr_ss_std, tt, pvalue, surr_ss_list    


def shuffle_target(target, threshold):
    
    idx_preictal = np.argwhere(target==1)
    idx_interictal = np.argwhere(target==0)
    
    # Define random preictal possible samples
    begin_sample = int(len(idx_preictal)*threshold) # moment when it is possible to start raising alarms
    end_sample = len(idx_interictal) - len(idx_preictal) # to avoid that random preictal overlaps defined preictal    
    
    # Define random positioning of preictal
    begin_random_preictal = random.choice(np.arange(begin_sample,end_sample))
    end_random_preictal = begin_random_preictal + len(idx_preictal)
    
    # Construct surrogates target
    shuffled_target = np.zeros(len(target), dtype=int)
    shuffled_target[begin_random_preictal:end_random_preictal] = 1
    
    return shuffled_target


def t_test(surr_ss, ss):

    # One-tailed t-test (left-hand/lower tail)
    # H0: surr_ss >= ss  (is rejected when p-value<0.05)
    # H1: surr_ss < ss
    
    tt, pvalue = stats.ttest_1samp(surr_ss, ss, alternative='less')

    return tt, pvalue