import matplotlib.pyplot as plt 
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from scipy import stats
import xlsxwriter as xw
import seaborn

patient_list = ['402','8902','11002','16202',
'21902','26102','30802',
'32702','45402','46702','50802',
'52302','53402','55202','56402',
'58602','60002','64702',
'75202','80702','85202','93402',
'93902','94402','95202','96002',
'98102','98202', '101702','102202',
'104602','109502','112802',
'113902','114702','114902','123902']

approaches = ['Approach A', 'Approach B', 'Approach C']

# %% --- Save the final training and testing results (best SOP) ---

data_SS = []
data_FPR = []
data_stat = []
data_n_stat = []
data_SOP = []

# Create xlsx for test results
path_wr_ts = 'Results/Final_test_results.xlsx'
wb_ts = xw.Workbook(path_wr_ts, {'nan_inf_to_errors': True})

# Create xlsx for train results
path_wr_tr = 'Results/Final_train_results.xlsx'
wb_tr = xw.Workbook(path_wr_tr, {'nan_inf_to_errors': True})


for approach in approaches:
    
    # --- Save training results --- 
    # add wroksheet for training results
    ws_tr = wb_tr.add_worksheet(approach)
    
    # Header format
    format_general = wb_tr.add_format({'bold':True, 'bg_color':'#F2F2F2'})
    format_train= wb_tr.add_format({'bold':True, 'bg_color':'#A8C2F6'})

    # Insert Header
    header_general = ['Patient']
    if approach != 'Approach A':
        header_train = ['SOP', 'Event', '#Features', 'Cost', 'SS samples', 'SP samples', 'Metric', 'Final Metric']
    else: 
        header_train = ['SOP', '#Features', 'Cost', 'SS samples', 'SP samples', 'Metric']

    row = 0
    col = 0
    ws_tr.write_row(row, col, header_general, format_general)
    col = len(header_general)
    ws_tr.write_row(row, col, header_train, format_train)
    row += 1
    col = 0
    
    # read the xlsx file with the training information for all SOPs
    path = f'../Results/{approach}/Train_results.xlsx'
    dict_df_train = pd.read_excel(path, None)
    sheet_name=dict_df_train.keys()
    
    best_SOP = []
    train_parameters = []
    
    for i in sheet_name:
           sheet_i = dict_df_train[i]
           
           if (approach == 'Approach A'):
               metric='Metric'
           else:
               metric='Final metric'
               
           # find the optimal SOP (the one with the highest training metric)
           best_SOP.append(int(sheet_i.loc[sheet_i[metric] == sheet_i[metric].max()]['SOP']))
           
           # get the training parameters and values of the best SOP
           i_max=sheet_i[metric].idxmax()
           if approach != 'Approach A':
               train_parameters.append(sheet_i.loc[i_max:i_max+2,'SOP':metric].values.tolist())
           else: 
               train_parameters.append(sheet_i.loc[i_max,'SOP':metric].values.tolist())
    
    # write the data in the worksheet
    for j in range(len(patient_list)):
        col = 0
        ws_tr.write(row, col, patient_list[j])
        col = 1
        if approach != 'Approach A':
            for data in train_parameters[j]:
                ws_tr.write_row(row, col, data)
                row += 1
        else:
            ws_tr.write_row(row, col, train_parameters[j])
            row += 1
    

    # --- Save testing results --- 
    final_data = [] 
    
    # add wroksheet for test results
    ws_test = wb_ts.add_worksheet(approach)
    
    # Header format
    format_general = wb_ts.add_format({'bold':True, 'bg_color':'#F2F2F2'})
    format_test = wb_ts.add_format({'bold':True, 'bg_color':'#A8C2F6'})

    # Insert Header
    header_general = ['Patient','SOP']
    header_test = ['#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','p-value', 'Statistically valid']

    row = 0
    col = 0
    ws_test.write_row(row, col, header_general, format_general)
    col = len(header_general)
    ws_test.write_row(row, col, header_test, format_test)
    row += 1
    col = 0
                       
    # read the xlsx file with the testing information for all SOPs
    path_test = f'../Results/{approach}/Test_results.xlsx'
    dict_df_test = pd.read_excel(path_test, None)
    sheet_name_test=dict_df_test.keys()
    
    # find and write the performances of the best SOP model
    performances = []
    
    for j in range(len(best_SOP)):
        
        sheet_name_j = 'SOP='+str(best_SOP[j])
        patient = patient_list[j]
        sheet_best_SOP = dict_df_test[sheet_name_j]
        
        if approach=='Approach A':
            patient_j=sheet_best_SOP.iloc[j]
        else:
            patient_j=sheet_best_SOP.iloc[j*4+3]
            
        performances_patient_j=patient_j.get(['#Predicted', '#False Alarms', 'SS', 'FPR/h','SS surrogate mean', 'SS surrogate std',  'p-value', 'Statistically valid' ])
        performances.append(performances_patient_j)
    
        data=list(performances[j])
        data.insert(0, patient_list[j])
        data.insert(1,best_SOP[j])
        final_data.append(data)
        
        ws_test.write_row(row, col, data)
        row += 1
        
    data_SOP.append(best_SOP)
    data_SS.append([j[4] for j in final_data])
    data_FPR.append([j[5] for j in final_data])
    data_n_stat.append(sum([j[9] for j in final_data])) 
    data_stat.append([j[9] for j in final_data])     

wb_ts.close()
wb_tr.close()

#%% -- Boxplots illustrating the SS and FPR/h values and bar plots with the number of validated patients for each approach --- 

performances_name=['SS', 'FPR/h', 'Nr patients validated']
fig, axs = plt.subplots(1,len(performances_name), figsize=(20, 9))
aprch_name=['Control', 'Chronological', 'Cumulative']

bp1=axs[0].boxplot(data_SS,  patch_artist=True, flierprops={'marker': 'o', 'markersize': 10})
bp2=axs[1].boxplot(data_FPR,  patch_artist=True, flierprops={'marker': 'o', 'markersize': 10})
bp3=axs[2].bar(aprch_name, data_n_stat, color='#E3797D')
bp=[bp1, bp2]

for i in range(len(performances_name)):
    #boxplots
    if i!=2:
        axs[i].set_ylabel(performances_name[i], fontsize=16)
        axs[i].axes.xaxis.set_ticklabels(aprch_name, fontsize=16)
        axs[i].axes.tick_params(axis='y', labelsize=16)
        for k in range(len(approaches)):
            bp[i]['medians'][k].set_color('black')
            bp[i]['medians'][k].set_linestyle('--')
            bp[i]['boxes'][k].set_facecolor('#82A69C')
    #barplots
    else:
        axs[i].set_ylabel(performances_name[i], fontsize=16)
        axs[i].axes.xaxis.set_ticklabels(aprch_name,fontsize=16)
        axs[i].axes.tick_params(axis='y', labelsize=16)
        axs[i].set_ylim([0,max(data_n_stat)+5])
        axs[i].bar_label(bp3, fontsize=16)
        
fig.subplots_adjust(  
    top=0.979,
    bottom=0.055,
    left=0.043,
    right=0.988,
    hspace=0.2,
    wspace=0.2)
       
# plt.savefig('Results/Comparision_approaches (Best_SOP).png')
# plt.close()

#%% --- Violin plots illustrating the SS, FPR/h, and SOP values and bar plots with the number of validated patients for each approach ---

my_pal = {"Control": "#f6d3b2", "Chronological": "#E3797D", "Cumulative": "#82A69C"}
fig, axs = plt.subplots(2, 2, figsize=(17,10))

#SS
df_SS = pd.DataFrame(np.array(data_SS).transpose(), columns=['Control', 'Chronological', 'Cumulative'])
sns.violinplot(ax=axs[0,0], data=df_SS, palette=my_pal,  linewidth=0.0, saturation = 1, inner=None, alpha=0.6)
sns.boxplot(ax=axs[0,0], data=df_SS, saturation=1, width=0.2, fill=True, palette=my_pal, medianprops=dict(linestyle='--'))
axs[0,0].set_ylabel("SS")

#FPR/h
df_FPR = pd.DataFrame(np.array(data_FPR).transpose(), columns=['Control', 'Chronological', 'Cumulative'])
sns.violinplot(ax=axs[0,1], data=df_FPR, palette=my_pal,  linewidth=0.0, saturation = 1, inner="stick", alpha=0.6)
sns.boxplot(ax=axs[0,1], data=df_FPR, saturation=1, width=0.2, fill=True, palette=my_pal, medianprops=dict(linestyle='--'))
axs[0,1].set_ylabel("FPR/h")

#SOP
df_SOP = pd.DataFrame(np.array(data_SOP).transpose(), columns=['Control', 'Chronological', 'Cumulative'])
sns.violinplot(ax=axs[1,0], data=df_SOP, palette=my_pal,  linewidth=0.0, saturation = 1, inner="stick", alpha=0.6)
sns.boxplot(ax=axs[1,0], data=df_SOP, saturation=1, width=0.2, fill=True, palette=my_pal, medianprops=dict(linestyle='--'))
axs[1,0].set_ylabel("SOP")

#validated patients
colors= ["#f6d3b2", "#E3797D", "#82A69C"]
plt_bar=axs[1,1].bar(aprch_name, data_n_stat, color=colors, alpha=0.6)
plt.bar_label(plt_bar)
axs[1,1].set_ylabel("Number of validated patients")

plt.subplots_adjust(
        top=0.93,
        bottom=0.1,
        left=0.075,
        right=0.935,
        hspace=0.265,
        wspace=0.27)

#plt.savefig('Results/performance_comparision.png')
#plt.close()

#%% --- Performance results for each patient and approach using a color gradient ---

app=['Crt', 'Chr', 'Cml']
df_SS = pd.DataFrame(data_SS, app, patient_list)
df_FPR = pd.DataFrame(data_FPR, app, patient_list)

data_stat = np.array(data_stat)
[row,column]=np.where(data_stat==1)
row = row + 0.5
column = column+0.5

fig, axs = plt.subplots(2, 1, figsize=(20, 4.5))

cbar_ax_1 = fig.add_axes([.94, 0.658, .015, .28])

heat_fig_1=sns.heatmap(df_SS, square=True, cmap=sns.light_palette('#6d8a82', as_cmap=True), cbar_ax=cbar_ax_1, cbar_kws={'label': 'Sensitivity', 'ticks' : [0, 0.5, 1]}, xticklabels = False, linecolor='white',linewidths=0.5, ax=axs[0])
heat_fig_1.axes.tick_params(axis='y', labelsize=16)
heat_fig_1.axes.tick_params(axis='x', labelsize=16)
axs[0].scatter(column, row, marker='d', s=70, color='w', edgecolors='k')
cbar_ax_1.set_ylabel('SS', size=16)
cbar_ax_1.tick_params(axis='y', labelsize=16)

cbar_ax_2 = fig.add_axes([.94, 0.293, .015, .28])
heat_fig_2=sns.heatmap(df_FPR, square=True, cmap=sns.light_palette('#E3797D', as_cmap=True), cbar_ax=cbar_ax_2, cbar_kws={'label': 'FPR/h', 'ticks' : [0, 5, 10]}, xticklabels = True, linecolor='white',linewidths=0.5, ax=axs[1])
heat_fig_2.axes.tick_params(axis='y', labelsize=16)
heat_fig_2.axes.tick_params(axis='x', labelsize=16)
cbar_ax_2.set_ylabel('FPR/h', size=16)
cbar_ax_2.tick_params(axis='y', labelsize=16)

fig.subplots_adjust(
top=0.98,
bottom=0.255,
left=0.025,
right=0.925,
hspace=0.0,
wspace=0.0)

#plt.savefig('Results/Overall_Performance.png')
#plt.close()