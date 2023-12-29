from import_data import import_data
from aux_fun import splitting
from testing_approachA  import main_testA
from testing_approachB  import main_testB
from testing_approachC  import main_testC
from plot_results import fig_performance, fig_performance_all_patient, fig_feature_selection_A, fig_feature_selection_B_C
from save_results import save_test_results_B_C, save_test_results_A


# %% --- Configurations ---

# Approach A --> controlo
# Approch B --> chronological firing power
# Approch C --> cumulative firing power

# patients_list = ['402','8902','11002','16202',
# '21902','26102','30802',
# '32702','45402','46702','50802',
# '52302','53402','55202','56402',
# '58602','60002','64702',
# '75202','80702','85202','93402',
# '93902','94402','95202','96002',
# '98102','98202', '101702','102202',
# '104602','109502','112802',
# '113902','114702','114902','123902']

patients_list = ['402','8902','11002','16202']

approach_list = ['Approach A', 'Approach B', 'Approach C']
 
# --- Train options ---
n_seizures_train = 3
SPH = 5 
SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

window_size=5
 
# %% Import Data | Siplitting Data | Test

for approach in approach_list:
    
    print(f'\n------------ {approach} ------------')
    
    information_general = {}
    information_train = {}
    information_test = {}
    final_information = []
    general_information = []
 
    for patient in patients_list:
        
        print(f'\n--------- Patient:{patient} ---------')

        # --- Import data ---
        features, datetimes, features_channels, feature_names, channel_names, seizure_info = import_data(patient)
        
        # --- Splitting data (train|test) ---
        data, datetimes, seizure_info = splitting(features, datetimes, seizure_info, n_seizures_train)   
        
        # --- Test ---
        info_te={}
        info_te['seizure']=seizure_info['test']
        info_te['features']=features_channels
        
        if approach == 'Approach A':
            info_test = main_testA(patient, approach, data['test'], datetimes['test'], info_te, SPH, window_size)
        elif approach == 'Approach B':
            info_test = main_testB(patient, approach, data['test'], datetimes['test'], info_te, SPH, window_size)
        elif approach == 'Approach C':
            info_test = main_testC(patient, approach, data['test'], datetimes['test'], info_te, SPH, window_size)
      
        general_information.append([patient, len(data['test']), SPH])
        information_test[patient] = info_test
        
        # --- Figure: performance (patient) ---
        fig_performance(patient, SOP_list, info_test, approach)
        
    # %% SAVE RESULTS
    
    # --- Figure: performance per SOP (all patient) ---
    fig_performance_all_patient(SOP_list, information_test, approach)
    
    # -- Save test results (excel) ---
    if approach=='Approach A':
        save_test_results_A(general_information, information_test, SOP_list, approach)
    else:   
        save_test_results_B_C(general_information, information_test, SOP_list, approach)
    
    # --- Figure: relative frequency of the selected features ---
    if approach=='Approach A':
        fig_feature_selection_A(patients_list, approach, feature_names, channel_names, features_channels)
    else:
        fig_feature_selection_B_C(patients_list, approach, feature_names, channel_names, features_channels)