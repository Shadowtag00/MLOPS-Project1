# !/usr/bin/env python
# coding: utf-8

# imports

import pandas as pd
import numpy as np
import os
import sklearn
import cProfile
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import hydra
from omegaconf import DictConfig
import prometheus_client
from prometheus_client import  Summary, Counter
import random
import time
import subprocess
#import webbrowser
# import seaborn as sns
# import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

import logging
from rich.logging import RichHandler
import sys


# Path to your Prometheus executable and configuration file
PROMETHEUS_PATH = 'prometheus'  # Update this to your actual Prometheus executable path
PROMETHEUS_CONFIG = 'prometheus.yml'  # Update this to your actual Prometheus configuration file path

# Webpage paths
#metrics = 'http://localhost:8000'
#PROMETHEUS_web = 'http://localhost:9090'

# Define Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
IMPORT_TIME = Summary('data_import_seconds', 'Time spent importing data')
CLEANING_TIME = Summary('data_cleaning_seconds', 'Time spent cleaning data')
TRAINING_TIME = Summary('model_training_seconds', 'Time spent training model')
PROCESSED_RECORDS = Counter('processed_records', 'Number of processed records')
MODEL_TRAINING_COUNT = Counter('model_training_count', 'Number of times model is trained')

#Log start
FORMAT = "%(message)s"
#logging.basicConfig(
#    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
#)
log = logging.getLogger("rich")
log.setLevel(logging.DEBUG)  # Set the log level

# Create handlers for stdout and stderr
stdout_handler = RichHandler(level=logging.INFO)
stderr_handler = RichHandler(level=logging.ERROR)

# Optionally, specify a format for log messages
formatter = logging.Formatter(FORMAT)
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# Add the handlers to the logger
log.addHandler(stdout_handler)
log.addHandler(stderr_handler)

# Example log messages
log.debug('This is a debug message')
log.info('This is an info message')
log.warning('This is a warning message')
log.error('This is an error message')
log.critical('This is a critical message')

@REQUEST_TIME.time()
def process_request(t):
    """A dummy function that takes some time."""
    time.sleep(t)

def start_prometheus():
    """Function to start Prometheus server."""
    return subprocess.Popen([PROMETHEUS_PATH, '--config.file', PROMETHEUS_CONFIG])




@IMPORT_TIME.time()
@hydra.main(version_base=None, config_path="conf", config_name=os.getenv("CONFIG", "monitoring"))
def main(cfg: DictConfig) -> None:

    prometheus_client.start_http_server(cfg.data.server_port)
    log.info(f"Prometheus metrics available on port {cfg.data.server_port}")
    # webbrowser.open(metrics)
    # webbrowser.open(PROMETHEUS_web)

    mlflow.set_tracking_uri("http://localhost:5000")
    
    log.debug("Data is being opened and processed")
    ccvi,COVstats,COVvacc,foodInsp,pop=importData()
    # Logging for data import
    log.debug("Data imported")
    log.debug("CCVI len:"+str(len(ccvi)))
    log.debug("COVstats len:" + str(len(COVstats)))
    log.debug("COVvacc len:" + str(len(COVvacc)))
    log.debug("foodINSP len:" + str(len(foodInsp)))
    log.debug("pop len:" + str(len(pop)))

    # Data preprocessing
    ccvi=cleanCCVI(ccvi)
    COVstats=cleanCOVIDStats(COVstats)
    passFail=cleanFoodInspection(foodInsp)
    pop=cleanPopulation(pop)
    COVvacc=cleanCOVIDVacc(COVvacc)
    mergedData=mergeData(COVstats,COVvacc,pop,passFail,ccvi)

    # Split training data
    # X_train, X_test, y_train, y_test=splitTrainingData(mergedData)
    lrMetrics, rfMetrics, gbrMetrics, svrMetrics=nFold(mergedData)
    logCrossValidationMetrics(lrMetrics, rfMetrics, gbrMetrics, svrMetrics)
    # MLflow experiment tracking
    # mlflow.set_experiment("Chicago Health Data Analysis")
    # with mlflow.start_run():
    #     # Model training and evaluation
    #     linearReg(X_train, X_test, y_train, y_test)
    #     randomForestRegression(X_train, X_test, y_train, y_test)
    #     gbr(X_train, X_test, y_train, y_test)
    #     svr(X_train, X_test, y_train, y_test)
    #
        # Log parameters and metrics to MLflow
        # mlflow.log_param("data_paths", cfg.data)
        # mlflow.log_artifact("prometheus.yml")
        #
        # # Log metrics for the experiment
        # mlflow.log_metric("processed_records", PROCESSED_RECORDS._value.get())
        # mlflow.log_metric("model_training_count", MODEL_TRAINING_COUNT._value.get())

    while cfg.data.loop:
        time.sleep(15)

def importData():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    current_file_dir = os.path.dirname(current_file_path)
    if (current_file_dir.__contains__("runner")):
        current_file_dir = os.path.join(os.path.dirname(current_file_dir), 'data')
    else:
        current_file_dir = os.path.join(current_file_dir, '..', 'data')

    # data_dir=os.getcwd()
    # data_dir =os.path.dirname(os.path.abspath(os.path.join(os.path.dirname("requirements.txt"), "..")))
    # data_dir=os.path.expanduser("~\\Documents\\MLOPS-Project1")
    # data_dir=os.path.join(data_dir, 'data')
    # Use the absolute paths to load the CSV files
    ccvi = pd.read_csv(os.path.join(current_file_dir, 'Chicago_COVID-19_Community_Vulnerability_Index__CCVI__-_ZIP_Code_Only.csv'))
    COVstats = pd.read_csv(os.path.join(current_file_dir, 'COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code.csv'))
    COVvacc = pd.read_csv(os.path.join(current_file_dir, 'COVID-19_Vaccinations_by_ZIP_Code_-_Historical.csv'))
    foodInsp = pd.read_csv(os.path.join(current_file_dir, 'Food_Inspections_20240322.csv'))
    pop = pd.read_csv(os.path.join(current_file_dir, 'Chicago_Population_Counts.csv'))
    return ccvi, COVstats, COVvacc, foodInsp, pop

# COVID 19 Stats Cleaning
@CLEANING_TIME.time()
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
    PROCESSED_RECORDS.inc(len(covstats_cleaned))
    log.debug("COVID-19 stats cleaned")
    log.debug("covstats_cleaned len:" + str(len(covstats_cleaned)))
    return covstats_cleaned

# COVID 19 Vaccinations Cleaning
@CLEANING_TIME.time()

def cleanCOVIDVacc(COVvacc):
    COVvacc.dropna(inplace=True)
    covdose_cleaned = COVvacc[['Zip Code', 'Total Doses - Daily']]
    covdose_cleaned.loc[:, 'Zip Code'] = covdose_cleaned['Zip Code'].astype('int64')
    # Group by 'ZIP Code' and calculate the sum of weekly cases and deaths, and the mean of weekly case rate
    aggregated_data_dose = covdose_cleaned.groupby('Zip Code').agg({
        'Total Doses - Daily': 'sum',
    }).reset_index()
    PROCESSED_RECORDS.inc(len(aggregated_data_dose))
    log.debug("COVID-19 Vaccination cleaned")
    log.debug("COVIDvacc_cleaned len:" + str(len(aggregated_data_dose)))
    return aggregated_data_dose

# CCVI Cleaning
# ccvi keep Community area or xip code, ccvi value, location(for now)
@CLEANING_TIME.time()
def cleanCCVI(ccvi):
    ccvi = ccvi[['Community Area or ZIP Code', 'CCVI Score']]
    PROCESSED_RECORDS.inc(len(ccvi))
    log.debug("covstats_cleaned len:" + str(len(ccvi)))
    return ccvi

# Food Inspections CLeaning
@CLEANING_TIME.time()
def cleanFoodInspection(foodInsp):
    foodInsp.dropna(inplace=True)
    # Create a boolean mask to filter out entries with specific result values
    mask = foodInsp['Results'].isin(['Pass', 'Fail'])
    # Apply the mask to filter out rows with specified result values
    filtered_foodInsp = foodInsp[mask]
    # Count the entries by the 'Results' column
    result_counts = filtered_foodInsp['Results'].value_counts()
    filtered_foodInsp = filtered_foodInsp[['Zip', 'Results']]
    filtered_foodInsp['Zip'] = filtered_foodInsp['Zip'].astype('int64')
    # Group by 'Zip Code' and 'Results' columns and count the entries
    counts_by_zip_results = filtered_foodInsp.groupby(['Zip', 'Results']).size()
    # Rename 'Pass w/ Conditions' to 'Pass'
    food_inspections_grouped = counts_by_zip_results.copy()
    # log.debug(food_inspections_grouped)
    # Calculate pass-to-fail ratio for each ZIP code
    pass_fail_ratio = food_inspections_grouped.loc[:, 'Pass'] / food_inspections_grouped.loc[:, 'Fail']
    pass_fail_ratio = pass_fail_ratio.reset_index()
    pass_fail_ratio.columns = ['Zip', 'Results']
    pass_fail_ratio['Results'] = pass_fail_ratio['Results'].fillna(0)
    # Print pass-to-fail ratio
    # print(pass_fail_ratio)
    PROCESSED_RECORDS.inc(len(pass_fail_ratio))
    log.debug("Food inspections grouped into pass/fail ratio by zip")
    log.debug("pass_fail len:" + str(len(pass_fail_ratio)))
    return pass_fail_ratio


 # Population Cleaning
@CLEANING_TIME.time()
def cleanPopulation(pop):
    # Create a boolean mask to filter out entries with specific 'Geography Type' values
    mask = pop['Geography Type'].isin(['Zip Code'])
    # Apply the mask to filter out rows with specified 'Geography Type' values
    filtered_pop = pop[mask]
    # Create a boolean mask to filter out entries with specific 'Year' values
    maskTwo = filtered_pop['Year'].isin([2021])
    # Apply the mask to filter out rows with specified 'Year' values
    filtered_pop = filtered_pop[maskTwo].reset_index(drop=True)
    # Select the relevant columns
    pop_final = filtered_pop[['Geography', 'Population - Total']]
    # Convert 'Geography' to numeric, coercing errors to NaN, then drop these rows
    pop_final['Geography'] = pd.to_numeric(pop_final['Geography'], errors='coerce')
    pop_final.dropna(subset=['Geography'], inplace=True)
    # Ensure 'Geography' is of type 'int64'
    pop_final['Geography'] = pop_final['Geography'].astype('int64')
    # Increment the processed records counter
    PROCESSED_RECORDS.inc(len(pop_final))
    # Log the cleaning process
    log.debug("Population cleaned")
    log.debug(f"pop_cleaned len: {len(pop_final)}")
    return pop_final


# Merge datasets on 'ZIP Code'
@CLEANING_TIME.time()
def mergeData(aggregated_data,aggregated_data_dose, pop_final,pass_fail_ratio,ccvi):
    merged_data = pd.merge(aggregated_data, aggregated_data_dose, left_on='ZIP Code', right_on='Zip Code', how='inner')
    merged_data = pd.merge(merged_data, pop_final, left_on='Zip Code', right_on='Geography', how='inner')
    merged_data = pd.merge(merged_data, pass_fail_ratio, left_on='Zip Code', right_on='Zip', how='inner')
    merged_data = pd.merge(merged_data, ccvi, left_on='ZIP Code', right_on='Community Area or ZIP Code', how='inner')
    # Drop unwanted columns
    merged_data.drop(columns=['Zip Code', 'Zip', 'Community Area or ZIP Code', 'Geography'], inplace=True)
    # Rename 'Cases - Weekly' column to 'Total COVID Cases'
    merged_data.rename(columns={'Cases - Weekly': 'Total COVID Cases'}, inplace=True)
    merged_data.rename(columns={'Deaths - Weekly': 'Total COVID Deaths'}, inplace=True)
    merged_data.rename(columns={'Total Doses - Daily': 'Total COVID Vacc Doses'}, inplace=True)
    merged_data.rename(columns={'Results': 'Food Insp: Pass/Fail ratio'}, inplace=True)
    PROCESSED_RECORDS.inc(len(merged_data))
    log.debug("Data Merged and filtered")
    log.debug("Merged list len:" + str(len(merged_data)))
    return merged_data

def nFold(mergedData):
    # Prepare the data for cross-validation
    X = mergedData.drop('Total COVID Deaths', axis=1)
    y = mergedData['Total COVID Deaths']
    lrMetrics = []
    rfMetrics = []
    gbrMetrics = []
    svrMetrics = []
    # Define the number of folds for cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # MLflow experiment tracking
    mlflow.set_experiment("Chicago Health Data Analysis")
    with mlflow.start_run():
        # Perform n-fold cross-validation for each model
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Model training and evaluation inside the cross-validation loop
            y_test,y_pred=linearReg(X_train, X_test, y_train, y_test)
            mae, mse, rmse=calculateMetrics(y_test, y_pred)
            mlflow.log_metric("lr_mae", mae)
            mlflow.log_metric("lr_mse", mse)
            mlflow.log_metric("lr_rmse", rmse)
            lrMetrics.append([mae,mse,rmse])
            y_test, y_pred =randomForestRegression(X_train, X_test, y_train, y_test)
            mae, mse, rmse = calculateMetrics(y_test, y_pred)
            mlflow.log_metric("rf_mae", mae)
            mlflow.log_metric("rf_mse", mse)
            mlflow.log_metric("rf_rmse", rmse)
            rfMetrics.append([mae, mse, rmse])
            y_test, y_pred = gbr(X_train, X_test, y_train, y_test)
            mae, mse, rmse = calculateMetrics(y_test, y_pred)
            mlflow.log_metric("gbr_mae", mae)
            mlflow.log_metric("gbr_mse", mse)
            mlflow.log_metric("gbr_rmse", rmse)
            gbrMetrics.append([mae, mse, rmse])
            y_test, y_pred = svr(X_train, X_test, y_train, y_test)
            mae, mse, rmse = calculateMetrics(y_test, y_pred)
            mlflow.log_metric("svr_mae", mae)
            mlflow.log_metric("svr_mse", mse)
            mlflow.log_metric("svr_rmse", rmse)
            svrMetrics.append([mae, mse, rmse])

    return lrMetrics, rfMetrics, gbrMetrics, svrMetrics

def logCrossValidationMetrics(lrMetrics, rfMetrics, gbrMetrics, svrMetrics):
    # Linear Regression Metrics
    mean, stdDev = calculateCrossValidationMetrics(lrMetrics)
    log.info(f"Linear Regression:\n"
             f"  Mean MAE: {mean[0]:.4f} | MAE Std Dev: {stdDev[0]:.4f}\n"
             f"  Mean MSE: {mean[1]:.4f} | MSE Std Dev: {stdDev[1]:.4f}\n"
             f"  Mean RMSE: {mean[2]:.4f} | RMSE Std Dev: {stdDev[2]:.4f}")

    # Random Forest Regression Metrics
    mean, stdDev = calculateCrossValidationMetrics(rfMetrics)
    log.info(f"Random Forest Regression:\n"
             f"  Mean MAE: {mean[0]:.4f} | MAE Std Dev: {stdDev[0]:.4f}\n"
             f"  Mean MSE: {mean[1]:.4f} | MSE Std Dev: {stdDev[1]:.4f}\n"
             f"  Mean RMSE: {mean[2]:.4f} | RMSE Std Dev: {stdDev[2]:.4f}")

    # Gradient Boost Regression Metrics
    mean, stdDev = calculateCrossValidationMetrics(gbrMetrics)
    log.info(f"Gradient Boost Regression:\n"
             f"  Mean MAE: {mean[0]:.4f} | MAE Std Dev: {stdDev[0]:.4f}\n"
             f"  Mean MSE: {mean[1]:.4f} | MSE Std Dev: {stdDev[1]:.4f}\n"
             f"  Mean RMSE: {mean[2]:.4f} | RMSE Std Dev: {stdDev[2]:.4f}")

    # Support Vector Regression Metrics
    mean, stdDev = calculateCrossValidationMetrics(svrMetrics)
    log.info(f"Support Vector Regression:\n"
             f"  Mean MAE: {mean[0]:.4f} | MAE Std Dev: {stdDev[0]:.4f}\n"
             f"  Mean MSE: {mean[1]:.4f} | MSE Std Dev: {stdDev[1]:.4f}\n"
             f"  Mean RMSE: {mean[2]:.4f} | RMSE Std Dev: {stdDev[2]:.4f}")

def calculateCrossValidationMetrics(performance_measures):
    # Calculate the mean and standard deviation for each performance measure
    mean = np.mean(performance_measures, axis=0)
    stdDev = np.std(performance_measures, axis=0)
    return mean, stdDev


# Models

@TRAINING_TIME.time()
def linearReg(X_train, X_test, y_train, y_test):
    MODEL_TRAINING_COUNT.inc()
    # Initialize the model
    model_lr = LinearRegression()
    # Train the model
    log.debug("Training model")
    model_lr.fit(X_train, y_train)
    # Predict on the test set
    log.debug("Testing model")
    y_pred_lr = model_lr.predict(X_test)
    # Calculate evaluation metrics
    log.debug("Linear Regression Completed")
    return y_test,y_pred_lr


@TRAINING_TIME.time()
def randomForestRegression(X_train, X_test, y_train, y_test):
    MODEL_TRAINING_COUNT.inc()
    # Initialize the model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model
    log.debug("Training model")
    model_rf.fit(X_train, y_train)
    # Predict on the test set
    log.debug("Testing model")
    y_pred_rf = model_rf.predict(X_test)
    log.debug("Random Forest Regression Completed")
    return y_test,y_pred_rf



@TRAINING_TIME.time()
def gbr(X_train, X_test, y_train, y_test):
    MODEL_TRAINING_COUNT.inc()
    # Initialize the model
    model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    # Train the model
    log.debug("Training model")
    model_gb.fit(X_train, y_train)
    # Predict on the test set
    log.debug("Testing model")
    y_pred_gb = model_gb.predict(X_test)
    log.debug("Gradient Boosting Regression Completed")
    return y_test,y_pred_gb


@TRAINING_TIME.time()
def svr(X_train, X_test, y_train, y_test):
    MODEL_TRAINING_COUNT.inc()
    # Initialize the model with StandardScaler
    model_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
    # Train the model
    log.debug("Training model")
    model_svr.fit(X_train, y_train)
    # Predict on the test set
    log.debug("Testing model")
    y_pred_svr = model_svr.predict(X_test)
    log.debug("SVR Completed")
    return y_test,y_pred_svr


def logResults(mae,mse,rmse):
    log.debug("Mean Absolute Error (MAE):" + str(mae)
             + ", Mean Squared Error (MSE):" + str(mse)
             + ", Root Mean Squared Error (RMSE):" + str(rmse))


def calculateMetrics(y_test, y_pred_svr):
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mse_svr)
    return mae_svr, mse_svr, rmse_svr

if __name__ == '__main__':

    log.info("Program Running")
   
    # Start Prometheus
    #prometheus_process = start_prometheus()
    log.info("Prometheus started")
    try:
        main()
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
       # prometheus_process.terminate()
        log.info("Prometheus terminated")


    

