import numpy as np
from datetime import timedelta as t

def get_firing_power(prediction, times, SOP, window_size):
    
    firing_power = []
    for i in range(len(prediction)):
        
        begin_time = times[i] - t(minutes = SOP)
        end_time = times[i]
        window_values = prediction[(times>begin_time) & (times<=end_time)]
        
        # Deal with time gaps
        if len(window_values)<int((SOP*60/window_size)/2):
            window_size_samples = int(SOP*60/window_size)
        else:
            window_size_samples = len(window_values)
        
        firing_power_value = np.sum(window_values)/window_size_samples
        firing_power.append(firing_power_value)
        
    firing_power = np.array(firing_power)
    
    return firing_power


def alarm_generation(firing_power, threshold):
    
    alarm = np.zeros(len(firing_power))
    for i in range(len(firing_power)):
        if firing_power[i] >= threshold:
            alarm[i] = 1
            
    return alarm
        

def alarm_processing(alarm, times, SOP, SPH):
    
    refractory_samples = np.full(len(alarm), False)   
    for i in range(len(alarm)):
        
        if alarm[i] == 1:
            
            # --- Refractory period (SOP + SPH) ---
            begin_time = times[i]
            end_time = times[i] + t(minutes = SOP + SPH)
            refractory_period = (times > begin_time) & (times <= end_time)
            
            # --- Next alarm can only be raised after the refractory period ---
            alarm[refractory_period] = 0

            # --- Refractory samples ---            
            refractory_samples[refractory_period] = True
                
    return alarm, refractory_samples

