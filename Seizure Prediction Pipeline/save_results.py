import numpy as np
import xlsxwriter as xw
import pickle
from aux_fun import get_selected_features


# --- Creates an xlsx file with the training results of approach A ---
def save_train_results_A(general_information, train_information, approach):
    
    # Create xlsx
    path = f'Results/{approach}/Train_results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})


    for patient in train_information:
        
        general_info = general_information.copy()
        general_info.insert(0,patient)
        train_info = train_information[patient]
        
        # Create sheet
        ws = wb.add_worksheet(f'pat_{patient}')

        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
        format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        
        # Insert Header
        header_general = ['Patient','#Seizures train','SPH','SOP']
        header_train = ['#Features','Cost','SS samples','SP samples','Metric']
           
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col = len(header_general)
        ws.write_row(row, col, header_train, format_train)
        
        # Insert data
        row = 1
        col = 0
        ws.write_row(row, col, general_info)
        
        col = len(general_info)
        for i in train_info:
            ws.write_row(row, col, i)
            row += 1

    wb.close()
    


# --- Creates an xlsx file with the training results of approach B and C ---
def save_train_results_B_C(general_information, train_information, approach):
    
    # Create xlsx
    path = f'Results/{approach}/Train_results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})


    for patient in train_information:
        
        general_info = general_information.copy()
        general_info.insert(0,patient)
        train_info = train_information[patient]
        
        # Create sheet
        ws = wb.add_worksheet(f'pat_{patient}')

        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
        format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        
        # Insert Header
        header_general = ['Patient','#Seizures train','SPH','SOP']
        header_train = ['Event','#Features','Cost','SS samples','SP samples','Metric']
        
        
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col = len(header_general)
        ws.write_row(row, col, header_train, format_train)
        
        # Insert data
        row = 1
        col = 0
        ws.write_row(row, col, general_info)
        
        col = len(general_info)
        for i in train_info:
            ws.write_row(row, col, i)
            row += 1

    wb.close()
    


# --- Creates an xlsx file with the testing results of approach B and C ---
def save_test_results_B_C(general_information, information_test, SOP_list, approach):
    
    # Create xlsx
    path = f'Results/{approach}/Test_results_new.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})
    
    i_test=information_test
    
    for SOP in SOP_list:
                
        # Create sheet
        ws = wb.add_worksheet(f'SOP={SOP}')
        
        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#bcbcbc'})
        format_event = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
        format_test = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        format_final = wb.add_format({'bg_color':'#F2F2F2'})
    
        # Insert Header
        header_general = ['Patient','#Seizures test','SPH']
        header_event = ['Event', 'SS samples', 'SP samples']
        header_test = ['#Predicted','#False Alarms','SS','FPR/h', 'SS surrogate mean','SS surrogate std','tt','p-value','Statistically valid']
                
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col += len(header_general)
        ws.write_row(row, col, header_event, format_event)
        col += len(header_event) 
        ws.write_row(row, col, header_test, format_test)
        row=1
        col=0
        
        n_patient=0
        for patient in i_test:
            # Insert data
            col=0
            ws.write_row(row, col, general_information[n_patient])
            col=len(general_information[n_patient])
            n_patient+=1
            
            info_test=i_test[patient][SOP]
            
            
            p_value=info_test[-1][-1]
            if p_value<0.05:
                stat=[1]
            else:
                stat=[0]
            
            for i in info_test:

                if i==info_test[-1]:    
                    i=['-','-','-']+info_test[-1].copy()+stat
                    ws.write_row(row, col, i, format_final)
                else:
                    ws.write_row(row, col, i)
                row+=1
    wb.close()
            
    
    
# --- Creates an xlsx file with the testing results of approach A ---
def save_test_results_A(general_information, information_test, SOP_list, approach):
    
    # Create xlsx
    path = f'Results/{approach}/Test_results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})
    
    i_test=information_test
    
    for SOP in SOP_list:
                
        # Create sheet
        ws = wb.add_worksheet(f'SOP={SOP}')
        
        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#bcbcbc'})
        format_event = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
        format_test = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
    
        # Insert Header
        header_general = ['Patient','#Seizures test','SPH']
        header_event = ['SS samples', 'SP samples']
        header_test = ['#Predicted','#False Alarms','SS','FPR/h', 'SS surrogate mean','SS surrogate std','tt','p-value','Statistically valid']
                
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col += len(header_general)
        ws.write_row(row, col, header_event, format_event)
        col += len(header_event) 
        ws.write_row(row, col, header_test, format_test)
        row=1
        col=0
        
        for j in general_information:
            # Insert data
            ws.write_row(row, col, j)
            row+=1
        
        row=1
        col=len(j)
        
        for patient in i_test:
            info_test=i_test[patient][SOP]
            
            p_value=info_test[-1]
            if p_value<0.05:
                stat=[1]
            else:
                stat=[0]
           
            i=info_test.copy()+stat
            
            ws.write_row(row, col, i)
            row+=1
            
    wb.close()