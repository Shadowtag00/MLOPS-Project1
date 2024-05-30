# !/usr/bin/env python
# coding: utf-8

# imports

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


# Dataset import


census = pd.read_csv('Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv')
ccvi = pd.read_csv('Chicago_COVID-19_Community_Vulnerability_Index__CCVI__-_ZIP_Code_Only.csv')
COVstats = pd.read_csv('COVID-19_Cases__Tests__and_Deaths_by_ZIP_Code.csv')
COVvacc =pd.read_csv('COVID-19_Vaccinations_by_ZIP_Code_-_Historical.csv')
foodInsp = pd.read_csv('Food_Inspections.csv')
PHStats = pd.read_csv('Public_Health_Statistics_-_Selected_public_health_indicators_by_Chicago_community_area_-_Historical.csv')
pop = pd.read_csv('Chicago_Population_Counts.csv')


# COVID 19 Stats Cleaning (1/6)

covstats_cleaned = COVstats[['ZIP Code', 'Cases - Weekly', 'Case Rate - Weekly', 'Deaths - Weekly']]
covstats_cleaned = covstats_cleaned[covstats_cleaned['ZIP Code'] != 'Unknown']
covstats_cleaned.dropna(inplace=True)
covstats_cleaned['ZIP Code'] = covstats_cleaned['ZIP Code'].astype('int64')
covstats_cleaned

# Group by 'ZIP Code' and calculate the sum of weekly cases and deaths, and the mean of weekly case rate
aggregated_data = covstats_cleaned.groupby('ZIP Code').agg({
    'Cases - Weekly': 'sum',
    'Deaths - Weekly': 'sum',
    'Case Rate - Weekly': lambda x: x.median() 
}).reset_index()



# COVID 19 Vaccinations Cleaning (2/6)

COVvacc.dropna(inplace=True)
covdose_cleaned = COVvacc[['Zip Code', 'Total Doses - Daily']]

covdose_cleaned['Zip Code'] = covdose_cleaned['Zip Code'].astype('int64')

# Group by 'ZIP Code' and calculate the sum of weekly cases and deaths, and the mean of weekly case rate
aggregated_data_dose = covdose_cleaned.groupby('Zip Code').agg({
    'Total Doses - Daily': 'sum',
}).reset_index()


# CCVI Cleaning (3/6)

# ccvi keep Community area or xip code, ccvi value, location(for now)
ccvi = ccvi[['Community Area or ZIP Code', 'CCVI Score', 'Location']]


# Food Inspections CLeaning (4/6)

# Drop null values
foodInsp.dropna(inplace=True)

# Create a boolean mask to filter out entries with specific result values
mask = foodInsp['Results'].isin(['Pass','Fail'])

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



# Public Health Stats Cleaning (5/6)

PHStats = PHStats.drop(columns=['Gonorrhea in Females','Gonorrhea in Males'])
PHStats.head()



# Population Cleaning (6/6)

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



# Merging data into one dataset

# Merge datasets on 'ZIP Code'
merged_data = pd.merge(aggregated_data, aggregated_data_dose,left_on='ZIP Code', right_on='Zip Code', how='inner')
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



# Split training data

from sklearn.model_selection import train_test_split

# Define X (features) and y (target)
X = merged_data.drop('Total COVID Deaths', axis=1)
y = merged_data['Total COVID Deaths']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))


# Models

from sklearn.linear_model import LinearRegression

# Initialize the model
model_lr = LinearRegression()

# Train the model
model_lr.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict on the test set
y_pred_lr = model_lr.predict(X_test)

# Calculate evaluation metrics
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print("Linear Regression Evaluation:")
print("Mean Absolute Error (MAE):", mae_lr)
print("Mean Squared Error (MSE):", mse_lr)
print("Root Mean Squared Error (RMSE):", rmse_lr)

print(y_pred_lr)

print(y_test.reset_index(drop=True, inplace=True))

print(y_test)


from sklearn.ensemble import RandomForestRegressor

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



from sklearn.ensemble import GradientBoostingRegressor

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



from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

