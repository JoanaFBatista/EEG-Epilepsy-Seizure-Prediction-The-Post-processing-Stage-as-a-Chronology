from import_data import import_data
from aux_fun import splitting, remove_SPH
import pandas
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md

#%% --- Plotting the Firing Power curve (Approach A) ---

def plot_FP_A(patient, approach, datetimes_test, onset_times, test_prediction, n_seizures_train):
    
    for SOP in test_prediction:
        
        target=test_prediction[SOP]['target']
        firing_power=test_prediction[SOP]['firing power']
        alarms=test_prediction[SOP]['alarm']
        
        # --- Load the vigilance vectors ---
        directory='Data/Vigilance_Vectors'
        vigilance=np.load(directory+"/pat_"+str(patient)+"_vigilance",allow_pickle=True)
        vigilance_datetimes=np.load(directory+"/pat_"+str(patient)+"_datetimes",allow_pickle=True)
        
        vigilance_timestamp=[]
        vigilance_time=[]
        for i in range(0,len(vigilance)):
            vigilance[i]=np.abs(vigilance[i]-1)
            vigilance[i]=np.clip(vigilance[i], 0.1, 0.9)
            
            times_stamp = np.array([time.timestamp() for time in vigilance_datetimes[i]])
            vigilance_timestamp.append(times_stamp)
            
            times_corr = np.array([pandas.to_datetime(time, unit='s') for time in vigilance_timestamp[i]])
            vigilance_time.append(times_corr)
    
        # --- Filling the missing data ---
        datetimes_new=[]
        firing_power_output_new=[]
        target_new=[]
        alarms_new=[]
        
        for i in range(0,len(datetimes_test)):
                   
            datetimes_new_i=[]
            firing_power_output_new_i=[]
            target_new_i=[]
            alarms_new_i=[]
            
            for j in range(0,len(datetimes_test[i])-1):
                time_difference=datetimes_test[i][j+1]-datetimes_test[i][j]
                time_difference=time_difference.seconds
            
                datetimes_new_i.append(datetimes_test[i][j])
                firing_power_output_new_i.append(firing_power[i][j])
                target_new_i.append(target[i][j])
                alarms_new_i.append(alarms[i][j])
                
                if time_difference<=5:
                    pass
                else:
                    new_datetime=datetimes_test[i][j]+datetime.timedelta(0,5)
                    while(time_difference>5):
                        datetimes_new_i.append(new_datetime)
                        target_new_i.append(np.NaN)
                        alarms_new_i.append(np.NaN)
                        firing_power_output_new_i.append(np.NaN)
                        
                        time_difference=datetimes_test[i][j+1]-new_datetime
                        time_difference=time_difference.seconds
                        new_datetime=new_datetime+datetime.timedelta(0,5)          
                        
            datetimes_new_i.append(datetimes_test[i][-1])
            firing_power_output_new_i.append(firing_power[i][-1])
            target_new_i.append(target[i][-1])
            alarms_new_i.append(alarms[i][-1])

            datetimes_new.append(datetimes_new_i)
            firing_power_output_new.append(firing_power_output_new_i)
            target_new.append(target_new_i)
            alarms_new.append(alarms_new_i)
    
    
        # --- Plotting Firing Power output throughout time ---  
        for i in range (0,len(datetimes_test)):
    
            plt.figure(figsize=(17,8)) 
            
            # plot final FP and FP threshold
            plt.plot(datetimes_new[i],firing_power_output_new[i],'k',alpha=0.7, label='Firing Power')
            plt.plot(datetimes_new[i],np.linspace(0.5, 0.5, len(datetimes_new[i])),linestyle='--',
                      color='black',alpha=0.7, label='Alarm threshold')
            plt.grid()
            plt.ylim(0,1)
            plt.xlim(datetimes_new[i][0],datetimes_new[i][len(datetimes_new[i])-1])
            
            # mark the predicted alarms    
            true_alarm=np.where(np.array(target_new[i])==1)[0]
            lb='False alarm'
            for alarm_index in np.where(np.array(alarms_new[i])==1)[0]:
                cl='maroon' #color for false alarms
                if alarm_index in true_alarm:
                    cl='green' #color for true alarms
                    lb='True alarm'
                plt.plot(datetimes_new[i][alarm_index], firing_power_output_new[i][alarm_index], marker='^', color=cl, markersize=22, label=lb)
                lb=''

            # color the spaces where the FP is above the threshold
            plt.fill_between(datetimes_new[i], 0.5, np.array(firing_power_output_new[i]), where=np.array(firing_power_output_new[i])>0.5,
                              facecolor='brown', alpha=0.5, label='FP above alarm threshold')
            
            # plot the vigilance states
            plt.plot(vigilance_time[i], vigilance[i], alpha=0.4, color='#E3797D', label='Vigilance state')
            
            # color the preictal period
            preictal=(np.array(target_new[i])==1)
            plt.fill_between(datetimes_new[i], 0, 1, where=preictal, facecolor='coral', alpha=0.5, label='Preictal period')
            i_preictal=datetimes_test[i][-1]-datetime.timedelta(minutes=SOP)
            plt.axvline(x = i_preictal, color = 'k', alpha = 0.7, linestyle='--',linewidth=0.8)
            
            #plot settings
            plt.legend(bbox_to_anchor =(0.5,-0.25), loc='lower center', ncol=8 , fontsize=12)
            plt.gcf().autofmt_xdate()
            xfmt = md.DateFormatter('%H:%M:%S')
            ax=plt.gca()
            ax.xaxis.set_major_formatter(xfmt)
            ax.yaxis.set_ticks([0,0.2,0.4,0.6,0.8,1.0])
            ax.yaxis.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax2= ax.twinx()
            ax2.yaxis.set_ticks([0.1, 0.9])
            ax2.yaxis.set_ticklabels(["sleep","awake"])
            ax2.tick_params(axis='y', labelsize=16)
            plt.title(f'Control\nPatient {patient}, Seizure {i+n_seizures_train+1}, SOP={SOP}', fontsize=16)
            
            plt.subplots_adjust(top=0.895,
                bottom=0.205,
                left=0.04,
                right=0.93,
                hspace=0.2,
                wspace=0.2)

            # plt.savefig(f'Results/FP({approach}, pat_{patient}, sz_{i+n_seizures_train+1}, SOP_{SOP}).png', dpi=330)
            # plt.close()  
  
            
#%% --- Plotting the Firing Power curve (Approach B/C) ---

def plot_FP_B_C(patient, approach, datetimes_test, onset_times, test_prediction, n_seizures_train):
    
    events=['Event 1', 'Event 2', 'Event 3']

    for SOP in test_prediction:
        
        # --- Load vigilance vectors --- 
        directory='Data/Vigilance_Vectors'
        vigilance=np.load(directory+"/pat_"+str(patient)+"_vigilance",allow_pickle=True)
        vigilance_datetimes=np.load(directory+"/pat_"+str(patient)+"_datetimes",allow_pickle=True)
        
        vigilance_timestamp=[]
        vigilance_time=[]
        for i in range(0,len(vigilance)):
            vigilance[i]=np.abs(vigilance[i]-1)
            vigilance[i]=np.clip(vigilance[i],0.1,0.9)
            
            times_stamp = np.array([time.timestamp() for time in vigilance_datetimes[i]])
            vigilance_timestamp.append(times_stamp)
            
            times_corr = np.array([pandas.to_datetime(time, unit='s') for time in vigilance_timestamp[i]])
            vigilance_time.append(times_corr)
    
                
        # --- Filling the missing data ---
        datetimes_new={}
        firing_power_output_new={}
        target_new={}
        
        for event in events:
            
            target=test_prediction[SOP][event]['target']
            firing_power=test_prediction[SOP][event]['firing power']
            
            datetimes_new_event=[]
            firing_power_output_new_event=[]
            target_new_event=[]
            
            for i in range(0,len(datetimes_test)):
            
                datetimes_new_event_i=[]
                firing_power_output_new_event_i=[]
                target_new_event_i=[]
                
                for j in range(0,len(datetimes_test[i])-1):
                    time_difference=datetimes_test[i][j+1]-datetimes_test[i][j]
                    time_difference=time_difference.seconds
                
                    datetimes_new_event_i.append(datetimes_test[i][j])
                    firing_power_output_new_event_i.append(firing_power[i][j])
                    target_new_event_i.append(target[i][j])
                    
                    if time_difference<=5:
                        pass
                    else:
                        new_datetime=datetimes_test[i][j]+datetime.timedelta(0,5)
                        while(time_difference>5):
                            datetimes_new_event_i.append(new_datetime)
                            target_new_event_i.append(np.NaN)
                            firing_power_output_new_event_i.append(np.NaN)
                            
                            time_difference=datetimes_test[i][j+1]-new_datetime
                            time_difference=time_difference.seconds
                            new_datetime=new_datetime+datetime.timedelta(0,5)
                
                datetimes_new_event_i.append(datetimes_test[i][-1])
                firing_power_output_new_event_i.append(firing_power[i][-1])
                target_new_event_i.append(target[i][-1])
                
                datetimes_new_event.append(datetimes_new_event_i)
                firing_power_output_new_event.append(firing_power_output_new_event_i)
                target_new_event.append(target_new_event_i)
            
            datetimes_new[event]=datetimes_new_event
            firing_power_output_new[event]=firing_power_output_new_event
            target_new[event]=target_new_event
            
            final_alarm=test_prediction[SOP]['final'][0] 
            final_alarm_per_seizure=[]
            final_alarm_new=[]
            
            i=0
            for sz in datetimes_test:
                f=i+len(sz)
                final_alarm_per_seizure.append(final_alarm[i:f])
                i=f

            for i in range(len(firing_power_output_new[event])):
                final_alarm_new_i=[]
                k=0
                fp_i=firing_power_output_new[event][i]
                for j in range(len(fp_i)):
                    if np.isnan(fp_i[j]):
                        final_alarm_new_i.append(np.NAN)
                    else:
                        final_alarm_new_i.append(final_alarm_per_seizure[i][k])
                        k+=1
                final_alarm_new.append(final_alarm_new_i)

        
        # --- Plotting Firing Power output throughout time ---   
        for i in range (0,len(datetimes_test)):
            
            fig, axs = plt.subplots(4, sharex=True, figsize=(17, 10))
            
            if approach == 'Approach B':
                fig.suptitle(f'Chronological Firing Power\nPatient {patient}, Seizure {i+n_seizures_train+1}, SOP={SOP}', fontsize=16)
            else:
                fig.suptitle(f'Cumulative Firing Power\nPatient {patient}, Seizure {i+n_seizures_train+1}, SOP={SOP}', fontsize=16)
            
            # --- Plotting final FP and FP threshold --- 
            label_fp='Firing Power'
            label_th='Alarm threshold'
            label_1='FP above alarm threshold'
            label_ev='Event'
            label_vg = 'Vigilance state'
            j=0
            for event in firing_power_output_new:
                
                # plot final FP and FP threshold
                axs[j].plot(datetimes_new[event][i],firing_power_output_new[event][i],'k',alpha=0.7, label=label_fp)
                axs[j].plot(datetimes_new[event][i],np.linspace(0.5, 0.5, len(datetimes_new[event][i])),linestyle='--',
                          color='black',alpha=0.7, label=label_th)
                axs[j].fill_between(datetimes_new[event][i], 0.5, np.array(firing_power_output_new[event][i]), where=np.array(firing_power_output_new[event][i])>0.5,
                                  facecolor='brown', alpha=0.5, label=label_1)
                
                # color the preictal events
                n=(np.array(target_new[event][i])==1)
                axs[j].fill_between(datetimes_new[event][i], 0, 1, where=n, facecolor='moccasin', alpha=0.5, label=label_ev)
                
                # plot the vigilance states
                axs[j].plot(vigilance_time[i], vigilance[i], alpha=0.4, color='#E3797D', label=label_vg)

                #plot settings
                axs[j].set_xlabel("", size=16)
                axs[j].set_ylabel(event, size=16)
                axs[j].grid()
                axs[j].set_ylim(0,1)
                axs[j].set_xlim(datetimes_new[event][i][0],datetimes_new[event][i][len(datetimes_new[event][i])-1])
                ax2= axs[j].twinx()
                ax2.yaxis.set_ticks([0.1, 0.9])
                ax2.yaxis.set_ticklabels(["sleep","awake"])
                ax2.tick_params(axis='y', labelsize=16)
                axs[j].yaxis.set_ticks([0, 0.5, 1.0])
                axs[j].tick_params(axis='y', labelsize=16)
                 
                label_fp = ''
                label_th = ''
                label_1 = ''
                label_ev = ''
                label_vg = ''
                j+=1
                
            axs[j].set_ylim(0,1)
            # Plot final Firing Power (Approach C)
            if approach=='Approach C':
                
                #calculate the final firing power
                fp=[]
                for event in firing_power_output_new:
                    fp.append(firing_power_output_new[event][i])
                final_firing_power_output=sum(np.array(fp))
                
                # plot the final firing power
                axs[j].plot(datetimes_new[event][i], final_firing_power_output,'k', alpha=0.7)
                axs[j].plot(datetimes_new[event][i], np.linspace(1.5, 1.5, len(datetimes_new[event][i])), linestyle='--', color='black', alpha=0.7)
                axs[j].fill_between(datetimes_new[event][i], 1.5, final_firing_power_output, where=final_firing_power_output>1.5, facecolor='brown', alpha=0.5)
                
                # plot settings
                axs[j].set_ylim(0,3)
                axs[j].yaxis.set_ticks([0, 1.5, 3])
                axs[j].tick_params(axis='y', labelsize=16)
                axs[j].grid()
                
            # --- Final Alarm ---
            # mark the predicted alarms    
            target_ev3=target_new['Event 3']
            true_alarm=np.where(np.array(target_ev3[i])==1)[0]
            lb='False Alarm'
            for alarm_index in np.where(np.array(final_alarm_new[i])==1)[0]:
                cl='maroon' #color for false alarms
                if alarm_index in true_alarm:
                    cl='green'  #color for true alarms
                    lb='True alarm'
                if approach=='Approach B':
                    axs[j].axvline(x = datetimes_new[event][i][alarm_index], ymin = 0, ymax = 0.5,  color=cl)
                    axs[j].plot(datetimes_new[event][i][alarm_index], 0.5, marker='^', color=cl, markersize=22, label=lb)
                    ylim_max=1
                else:
                    axs[j].plot(datetimes_new[event][i][alarm_index], final_firing_power_output[alarm_index], marker='^', color=cl, markersize=22, label=lb)
                    ylim_max=3
                lb=''
            
            # color the preictal period
            n=(np.array(target_new[event][i])==1)
            axs[j].fill_between(datetimes_new[event][i], 0, ylim_max, where=n, facecolor='coral', alpha=0.5, label='Preictal period')
           
            #plot settings
            axs[j].set_xlabel("", size=24)
            axs[j].set_ylabel('Final alarm', size=16)
            if approach=='Approach B':
                axs[j].axes.yaxis.set_ticklabels([])
                axs[j].tick_params(left = False, bottom = False)
                axs[j].grid(axis='x')
            plt.gcf().autofmt_xdate()
            xfmt = md.DateFormatter('%H:%M:%S')
            ax=plt.gca()
            ax.xaxis.set_major_formatter(xfmt)
            axs[j].tick_params(axis='x', labelsize=14)
            fig.legend(loc='lower center', ncol=8, fontsize=12)
    
            plt.subplots_adjust(top=0.925,
                bottom=0.13,
                left=0.065,
                right=0.93,
                hspace=0.2,
                wspace=0.2)
            
            # plt.savefig(f'Results/FP({approach}, pat_{patient}, sz_{i+n_seizures_train+1}, SOP_{SOP}).png', dpi=330)
            # plt.close()
        
#%% MAIN

patient_list=['402','8902','11002','16202',
'21902','26102','30802',
'32702','45402','46702','50802',
'52302','53402','55202','56402',
'58602','60002','64702',
'75202','80702','93402',
'93902','94402','95202','96002',
'98102','98202', '101702','102202',
'104602','109502','112802',
'113902','114702','114902','123902']

approach_list=['Approach A', 'Approach B', 'Approach C']

n_seizures_train=3
SPH=5

for patient in patient_list:
    
    print(f'\n--------- Patient:{patient} ---------')
    
    # --- Import data ---
    features, datetimes, features_channels, feature_names, channel_names, seizure_info = import_data(patient)
    
    # --- Splitting data (train|test) ---
    data, datetimes, seizure_info=splitting(features, datetimes, seizure_info, n_seizures_train)   
    
    # --- Remove SPH ---
    data_list_test = [remove_SPH(data['test'][seizure], datetimes['test'][seizure], SPH, seizure_info['test'][seizure]) for seizure in range(0,len(data['test']))]
    times_list_test = [datetimes['test'][seizure][:data_list_test[seizure].shape[0]] for seizure in range(0,len(data['test']))]
       
    # --- Onset time seizures ---
    onset_time_train=[seizure_info['train'][i][0] for i in range(0,len(seizure_info['train']))]
    onset_time_test=[seizure_info['test'][i][0] for i in range(0,len(seizure_info['test']))]
    onset_time={}
    onset_time['train'] = np.array([pandas.to_datetime(time, unit='s') for time in onset_time_train])
    onset_time['test'] = np.array([pandas.to_datetime(time, unit='s') for time in onset_time_test])

    
    for approach in approach_list:
        
        # --- Load test predictions ---
        with open(f'../Results/{approach}/Patient {patient}/test_prediction_patient{patient}', 'rb') as test_prediction_patient:
            test_prediction = pickle.load(test_prediction_patient)
        
        if approach=='Approach A':
            legend = plot_FP_A(patient, approach, times_list_test, onset_time, test_prediction, n_seizures_train)
        else:
            plot_FP_B_C(patient, approach, times_list_test, onset_time, test_prediction, n_seizures_train)
             