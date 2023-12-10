# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:30:56 2023

@author: hj
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split  # Splitting dataset
# Cross-validation and hyperparameter optimization
import optuna 
from sklearn.model_selection import cross_validate, KFold 
# SHAP model explanation method
import shap
# Functions for model performance evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # MSE, MAE, R2
def RMSE(y, y_pred):
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    return RMSE

def MAPE(y, y_pred):
    e = np.abs((y_pred - y) / y)
    MAPE = np.sum(e) / len(e)
    return MAPE

# Hyperparameter Tuning and Model Optimization with Bayesian Optimization
# Step 1: Define the Objective Function and Parameter Space
def optuna_objective(trial): 
    # Define the parameter space
    max_depth = trial.suggest_int("max_depth", 8, 12, 1)
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 400, 700, 50)
    subsample = trial.suggest_float("subsample", 0.5, 1, log=False)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1, log=False)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 4, 1)
    reg_lambda = trial.suggest_int("reg_lambda", 6, 12, 1)

    # Define the estimator
    reg = XGBRegressor(max_depth=max_depth, 
                       learning_rate=learning_rate,      
                       n_estimators=n_estimators,
                       objective='reg:squarederror', 
                       booster='gbtree', 
                       gamma=0,
                       min_child_weight=min_child_weight,
                       subsample=subsample, 
                       colsample_bytree=colsample_bytree,       
                       reg_alpha=0,
                       reg_lambda=reg_lambda
                       )   
    # 5-fold cross-validation process, output negative root mean squared error (-RMSE)
    cv = KFold(n_splits=5, shuffle=True)
    validation_loss = cross_validate(reg, x_train, y_train,
                                     scoring="neg_root_mean_squared_error",
                                     cv=cv,  # Cross-validation mode
                                     verbose=False,  # Print process
                                     n_jobs=-1,  # Number of threads
                                     error_score='raise'
                                     )
    # Final output: RMSE
    return np.mean(abs(validation_loss["test_score"]))



# Step 2: Define the Specific Workflow for Optimizing the Objective Function
def optimizer_optuna(n_trials, algo):
    # Define the sampler
    algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    
    # Create an Optuna study for optimization
    study = optuna.create_study(sampler=algo, direction="minimize")
    
    # Optimize the objective function
    study.optimize(optuna_objective,  # Objective function
                   n_trials=n_trials,   # Maximum number of iterations (including initial observations)
                   show_progress_bar=True  # Show progress bar
                  )
    
    # Display the best parameters and score
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")
    
    return study.best_trial.params, study.best_trial.values


# Read the data
dataset = pd.read_excel('D:\data.xlsx', 'dataset', index_col=None, keep_default_na=True)
# Extract input features, DD model or DKD model
'''DD model'''
# x_with_name = dataset.iloc[:, 3:21]
'''DKD model'''
x_with_name = dataset.iloc[:, np.r_[3:21]]
x = x_with_name.to_numpy(dtype=np.float64)
y = dataset.iloc[:, 25:26].to_numpy(dtype=np.float64)

'''Random splitting'''
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=290)

'''Hyperparameter optimization'''
param = {}
param, best_score = optimizer_optuna(30, "TPE")

''' Model developing'''
model = XGBRegressor(
    max_depth=param['max_depth'],
    learning_rate=param['learning_rate'],
    n_estimators=param['n_estimators'],
    objective='reg:squarederror',
    booster='gbtree',
    gamma=0.2,
    min_child_weight=param['min_child_weight'],
    subsample=param['subsample'],
    colsample_bytree=param['colsample_bytree'],
    reg_alpha=0,
    reg_lambda=param['reg_lambda']
)  
       

model.fit(x_train,y_train, eval_set=[(x_test,y_test)],
          eval_metric ='rmse',early_stopping_rounds = 20,
          verbose = True)

'''Model performance evaluation'''
y_pred_train = model.predict(x_train)
y_pred_train = np.reshape(y_pred_train, (len(y_pred_train), 1))
y_pred_train = np.where(y_pred_train > 100, 100, y_pred_train)
y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
MAE_train = mean_absolute_error(y_train, y_pred_train)
print("Training Set MAE: %.2f%%" % (MAE_train))
RMSE_train = RMSE(y_train, y_pred_train)
print("Training Set RMSE: %.2f%%" % (RMSE_train))
R2_train = r2_score(y_train, y_pred_train)
print("Training Set R2: %.4f" % (R2_train))

y_pred_test = model.predict(x_test)
y_pred_test = np.reshape(y_pred_test, (len(y_pred_test), 1))
y_pred_test = np.where(y_pred_test > 100, 100, y_pred_test)
y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
MAE_test = mean_absolute_error(y_test, y_pred_test)
print("Testing Set MAE: %.2f%%" % (MAE_test))
RMSE_test = RMSE(y_test, y_pred_test)
print("Testing Set RMSE: %.2f%%" % (RMSE_test))
R2_test = r2_score(y_test, y_pred_test)
print("Testing Set R2: %.4f" % (R2_test))

'''SHAP interpretability method'''
explainer = shap.TreeExplainer(model) # Initialize the explainer
expected_value = explainer.expected_value # Calculate the baseline value for the entire sample
shap_values = explainer.shap_values(x_with_name) # Calculate SHAP values for each feature of each sample
shap_explanation = shap.Explanation(shap_values, data=x, feature_names=np.array(x_with_name.columns))

'''Global importance ranking plot, indicating the overall importance of each feature (agnostic to positive or negative, averaging all SHAP values)'''
shap.summary_plot(shap_values, x_with_name, plot_type="bar", show=True)
shap.bar_plot(shap_values=shap_values[1], feature_names=x_with_name.columns)

'''Summary plot of feature importance (with positive and negative effects), requires further styling optimization, looks too unattractive'''
shap.summary_plot(shap_values, x_with_name, show=True)


'''Single Sample Decision Process Visualization: Force Plot and Waterfall Plot'''
# shap.initjs() # Initialize JS
# shap.force_plot(explainer.expected_value, shap_values[11], x_with_name.iloc[11], matplotlib=True, show=True) # Index 11 represents the sample's index in the dataset
# plt.tight_layout()

'''The waterfall plot is designed to illustrate the explanation of a single prediction, taking a single row of the explanation object as input. It shows how values are pushed from the model's expected output on the dataset to the predicted output.'''
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[1], feature_names=x_with_name.columns)
# plt.tight_layout()

'''Multi-Sample Decision Process Visualization: SHAP Decision Plot'''
# # Example: Select samples for analysis
# Samples = x_with_name.iloc[range(110, 130)]
# ys = y[range(110, 130)]
# # Filtering typical misclassified samples using a boolean one-dimensional array
# shap_values = explainer.shap_values(Samples)
# y_pred = shap_values.sum(1) + expected_value
# y_pred = y_pred.reshape(len(y_pred), 1)
# misclassified = abs(y_pred - ys) > 20
# misclassified = misclassified.reshape(len(misclassified),)
# shap.decision_plot(expected_value, shap_values, Samples, highlight=misclassified)
# plt.tight_layout()
# # Note: The decision plot is clearer and more intuitive than the force plot, especially when analyzing many features. By comparing the prediction process of typical misclassified samples with correct samples, it can explain the reasons for the errors.。


'''Partical Dependence Plots'''
feature_names = [
    r'$\mathrm{Pure\ water\ flux\ (L\cdot m^{-2}\cdot h^{-1})}$',
    "Pressure (bar)",
    "pH",
    "Temperature (°C)",
    "Filtration duration (h)",
    "TrOC concentration (mg/L)",
    "Cross-flow velocity (cm/s)",
    "MW (Da)",
    "MWCO (Da)",
    'Min projection (nm)',
    'Max projection (nm)',
    'Molecular radius (nm)',
    'Pore radius (nm)',
    r'$\mathrm{p}K_\mathrm{a1}$',
    r'$\mathrm{p}K_\mathrm{a2}$',
    'Zeta potential (mV)',
    r'$\mathrm{log }K_\mathrm{ow}$',
    'Contact angle (°)',
    'ɸ',
    'Molecular charge',
    'Charge product',
    r'$\mathrm{log }D$'
]

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'

save_path = "D:\\results\\"
for i, name in enumerate(feature_names):
    plt.rcParams['axes.unicode_minus'] = False  # To display negative sign properly
    shap.dependence_plot(name, shap_values, x_with_name, feature_names=feature_names, show=False)
    plt.xticks(fontsize=12)  # Set x-axis font size
    plt.yticks(fontsize=12)  # Set y-axis font size
    plt.ylabel('SHAP value', fontsize=15)  # Set y-axis label size
    plt.xlabel(name, fontsize=15)  # Set x-axis label size
    plt.tight_layout()  # Ensure coordinates are fully displayed
    column_name = x_with_name.columns[i]  # Get the name of the i-th column in x_with_name
    file_name = re.sub(r'[<>:"/\\|?*\x00-\x1F\x7F]', '', column_name) + '.png'  # Replace spaces with underscores and add .png suffix
    plt.savefig(save_path + file_name, dpi=1000)  # Save the image to the specified path

# Additional Note: Ensure that SHAP values are properly interpreted for univariate dependence analysis.