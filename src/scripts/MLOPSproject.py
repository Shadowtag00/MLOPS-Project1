# !/usr/bin/env python
# coding: utf-8

# imports

import pandas as pd
import numpy as np
import sklearn
import cProfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.profiler import profile, record_function, ProfilerActivity

# import seaborn as sns
# import matplotlib.pyplot as plt


# Dataset import
def importData():
    ccvi = pd.read_csv('data/Chicago_COVID-19_Community_Vulnerability_Index__CCVI__-_ZIP_Code_Only.csv')
    COVstats = pd.read_csv('data/COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code.csv')
    COVvacc = pd.read_csv('data/COVID-19_Vaccinations_by_ZIP_Code_-_Historical.csv')
    foodInsp = pd.read_csv('data/Food_Inspections_20240322.csv')
    pop = pd.read_csv('data/Chicago_Population_Counts.csv')
    return ccvi,COVstats,COVvacc,foodInsp,pop

# COVID 19 Stats Cleaning (1/6)
def cleanCOVIDStats(COVstats):
    covstats_cleaned = COVstats[['ZIP Code', 'Cases - Weekly', 'Case Rate - Weekly', 'Deaths - Weekly']]
    covstats_cleaned = covstats_cleaned[covstats_cleaned['ZIP Code'] != 'Unknown']
    covstats_cleaned.dropna(inplace=True)
    covstats_cleaned['ZIP Code'] = covstats_cleaned['ZIP Code'].astype('int64')
    covstats_cleaned = covstats_cleaned.groupby('ZIP Code').agg({
        'Cases - Weekly': 'sum',
        'Deaths - Weekly': 'sum',
        'Case Rate - Weekly': lambda x: x.median()
    }).reset_index()
    return covstats_cleaned

# COVID 19 Vaccinations Cleaning (2/6)
def cleanCOVIDVacc(COVvacc):
    COVvacc.dropna(inplace=True)
    covdose_cleaned = COVvacc[['Zip Code', 'Total Doses - Daily']]

    covdose_cleaned['Zip Code'] = covdose_cleaned['Zip Code'].astype('int64')

    # Group by 'ZIP Code' and calculate the sum of weekly cases and deaths, and the mean of weekly case rate
    aggregated_data_dose = covdose_cleaned.groupby('Zip Code').agg({
        'Total Doses - Daily': 'sum',
    }).reset_index()
    return aggregated_data_dose

# CCVI Cleaning (3/6)

# ccvi keep Community area or xip code, ccvi value, location(for now)
def cleanCCVI(ccvi):
    ccvi = ccvi[['Community Area or ZIP Code', 'CCVI Score', 'Location']]
    return ccvi

# Food Inspections CLeaning (4/6)

# Drop null values
def cleanFoodInspection(foodInsp):
    foodInsp.dropna(inplace=True)

    # Create a boolean mask to filter out entries with specific result values
    mask = foodInsp['Results'].isin(['Pass', 'Fail'])

    # Apply the mask to filter out rows with specified result values
    filtered_foodInsp = foodInsp[mask]

    # Count the entries by the 'Results' column
    result_counts = filtered_foodInsp['Results'].value_counts()

    # Print the result counts
    print(result_counts)

    filtered_foodInsp = filtered_foodInsp[['Zip', 'Results', 'Location']]
    filtered_foodInsp['Zip'] = filtered_foodInsp['Zip'].astype('int64')
    #filtered_foodInsp

    # Group by 'Zip Code' and 'Results' columns and count the entries
    counts_by_zip_results = filtered_foodInsp.groupby(['Zip', 'Results']).size()

    # Print the counts
    print(counts_by_zip_results)

    # Rename 'Pass w/ Conditions' to 'Pass'
    food_inspections_grouped = counts_by_zip_results.copy()
    print(food_inspections_grouped)

    # Calculate pass-to-fail ratio for each ZIP code
    pass_fail_ratio = food_inspections_grouped.loc[:, 'Pass'] / food_inspections_grouped.loc[:, 'Fail']
    pass_fail_ratio = pass_fail_ratio.reset_index()
    pass_fail_ratio.columns = ['Zip', 'Results']
    pass_fail_ratio['Results'] = pass_fail_ratio['Results'].fillna(0)

    # Print pass-to-fail ratio
    print(pass_fail_ratio)
    return pass_fail_ratio
# Population Cleaning (6/6)
def cleanPopulation(pop):
    # Create a boolean mask to filter out entries with specific result values
    mask = pop['Geography Type'].isin(['Zip Code'])

    # Apply the mask to filter out rows with specified result values
    filtered_pop = pop[mask]

    # Create a boolean mask to filter out entries with specific result values
    maskTwo = filtered_pop['Year'].isin([2021])

    # Apply the mask to filter out rows with specified result values
    filtered_pop = filtered_pop[maskTwo].reset_index()

    pop_final = filtered_pop[['Geography', 'Population - Total']]
    pop_final['Geography'] = pop_final['Geography'].astype('int64')
    return pop_final
# Merging data into one dataset

# Merge datasets on 'ZIP Code'
def mergeData(aggregated_data,aggregated_data_dose, pop_final,pass_fail_ratio,ccvi):
    merged_data = pd.merge(aggregated_data, aggregated_data_dose, left_on='ZIP Code', right_on='Zip Code', how='inner')
    merged_data = pd.merge(merged_data, pop_final, left_on='Zip Code', right_on='Geography', how='inner')
    merged_data = pd.merge(merged_data, pass_fail_ratio, left_on='Zip Code', right_on='Zip', how='inner')
    merged_data = pd.merge(merged_data, ccvi, left_on='ZIP Code', right_on='Community Area or ZIP Code', how='inner')

    # Drop unwanted columns
    merged_data.drop(columns=['Zip Code', 'Zip', 'Community Area or ZIP Code', 'Location', 'Geography'], inplace=True)

    # Rename 'Cases - Weekly' column to 'Total COVID Cases'
    merged_data.rename(columns={'Cases - Weekly': 'Total COVID Cases'}, inplace=True)
    merged_data.rename(columns={'Deaths - Weekly': 'Total COVID Deaths'}, inplace=True)
    merged_data.rename(columns={'Total Doses - Daily': 'Total COVID Vacc Doses'}, inplace=True)
    merged_data.rename(columns={'Results': 'Food Insp: Pass/Fail'}, inplace=True)

    print(merged_data)
    print(merged_data.info())
    print(merged_data.describe())
    return merged_data
# Split training data
def splitTrainingData(merged_data):

    # Define X (features) and y (target)
    X = merged_data.drop('Total COVID Deaths', axis=1)
    y = merged_data['Total COVID Deaths']

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))
    return X_train, X_test, y_train, y_test

# Models


def linearReg(X_train, X_test, y_train, y_test):
    # Initialize the model
    model_lr = LinearRegression()

    # Profile the training of the model
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("train_model"):
            # Train the model
            model_lr.fit(X_train, y_train)

    # Profile the prediction step
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("lin-reg"):
            # Predict on the test set
            y_pred_lr = model_lr.predict(X_test)

    # Calculate evaluation metrics
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)

    # Print the profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Print evaluation metrics
    print("Linear Regression Evaluation:")
    print("Mean Absolute Error (MAE):", mae_lr)
    print("Mean Squared Error (MSE):", mse_lr)
    print("Root Mean Squared Error (RMSE):", rmse_lr)

    # Print predictions
    print(y_pred_lr)

    # Reset the index of y_test for comparison
    y_test.reset_index(drop=True, inplace=True)

    # Print the true values
    print(y_test)

# Example usage
# X_train, X_test, y_train, y_test = your_data_loading_function()
# linearReg(X_train, X_test, y_train, y_test)

# def linearReg(X_train, X_test, y_train, y_test):
#     # Initialize the model
#     model_lr = LinearRegression()
#
#     # Train the model
#     model_lr.fit(X_train, y_train)
#
#
#
#     # Predict on the test set
#     y_pred_lr = model_lr.predict(X_test)
#
#     # Calculate evaluation metrics
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     rmse_lr = np.sqrt(mse_lr)
#
#     print("Linear Regression Evaluation:")
#     print("Mean Absolute Error (MAE):", mae_lr)
#     print("Mean Squared Error (MSE):", mse_lr)
#     print("Root Mean Squared Error (RMSE):", rmse_lr)
#
#     print(y_pred_lr)
#
#     print(y_test.reset_index(drop=True, inplace=True))
#
#     print(y_test)

def randomForestRegression(X_train, X_test, y_train, y_test):
    # Initialize the model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model_rf.fit(X_train, y_train)

    # Predict on the test set
    y_pred_rf = model_rf.predict(X_test)

    # Calculate evaluation metrics
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)

    print("Random Forest Regression Evaluation:")
    print("Mean Absolute Error (MAE):", mae_rf)
    print("Mean Squared Error (MSE):", mse_rf)
    print("Root Mean Squared Error (RMSE):", rmse_rf)

def gbr(X_train, X_test, y_train, y_test):
    # Initialize the model
    model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    model_gb.fit(X_train, y_train)

    # Predict on the test set
    y_pred_gb = model_gb.predict(X_test)

    # Calculate evaluation metrics
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mse_gb)

    print("Gradient Boosting Regression Evaluation:")
    print("Mean Absolute Error (MAE):", mae_gb)
    print("Mean Squared Error (MSE):", mse_gb)
    print("Root Mean Squared Error (RMSE):", rmse_gb)


    # Initialize the model with StandardScaler
    model_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

    # Train the model
    model_svr.fit(X_train, y_train)

    # Predict on the test set
    y_pred_svr = model_svr.predict(X_test)

    # Calculate evaluation metrics
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mse_svr)

    print("Support Vector Regression (SVR) Evaluation:")
    print("Mean Absolute Error (MAE):", mae_svr)
    print("Mean Squared Error (MSE):", mse_svr)
    print("Root Mean Squared Error (RMSE):", rmse_svr)


if __name__ == '__main__':
    ccvi,COVstats,COVvacc,foodInsp,pop=importData()
    ccvi=cleanCCVI(ccvi)
    COVstats=cleanCOVIDStats(COVstats)
    passFail=cleanFoodInspection(foodInsp)
    pop=cleanPopulation(pop)
    COVvacc=cleanCOVIDVacc(COVvacc)
    mergedData=mergeData(COVstats,COVvacc,pop,passFail,ccvi)
    X_train, X_test, y_train, y_test=splitTrainingData(mergedData)
    linearReg(X_train, X_test, y_train, y_test)
    randomForestRegression(X_train, X_test, y_train, y_test)
    gbr(X_train, X_test, y_train, y_test)