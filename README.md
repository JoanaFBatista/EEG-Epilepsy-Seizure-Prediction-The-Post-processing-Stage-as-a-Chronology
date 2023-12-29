# EEG-Epilepsy-Seizure-Prediction-The-Post-processing-Stage-as-a-Chronology
Code used to develop the methodologies proposed in "EEG Epilepsy Seizure Prediction: The Post-processing Stage as a Chronology".

Scripts used to develop the methodology proposed in "EEG Epilepsy Seizure Prediction: The Post-processing Stage as a Chronology". 

You can not execute these codes as it is necessary the preprocessed data from EEG recordings. As the used dataset belongs to EPILEPSIAE, we can not make it publicly available online due to ethical concerns.

## Seizure Prediction Pipeline
- [main_training.py] - execute it to train the model and to get the best grid-search parameters (preictal period, k number of features, SVM C value).
- [main testing.py] - execute it to test the model in new seizures, get the performance (seizure sensitivity, FPR/h, and surrogate analysis), and get the selected features.

- [aux_fun.py] - code with utility functions.
- [import_data.py] - code to import data.
- [regularization.py] - code to perform the regularization step using the Firing Power method.
- [training.py] - code to execute the grid search and to train the model.
- [testing_ApproachA.py] - code to test and evaluate the model considering the Control approach.
- [testing_ApproachB.py] - code to test and evaluate the model considering the Chronological Firing Power approach.
- [testing_ApproachC.py] - code to test and evaluate the model considering the Cumulative Firing Power approach.
- [save_results.py] - code to save the results for each patient in xlsx files.
- [plot_results.py] - code to get performance plots and selected features. 
- [evaluation.py] - code to evaluate the tested models (seizure sensitivity, FPR/h, and surrogate analysis).
- [plot_FP.py] - code to get the Firing Power plots.
- [plot_results_bestSOP] - code to save the results in xlsx files and to get performance plots considering the optimal SOP.


