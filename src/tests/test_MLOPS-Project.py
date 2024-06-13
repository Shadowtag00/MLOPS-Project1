# !/usr/bin/env python

# Example Tests

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.scripts.MLOPSproject import cleanCOVIDStats, cleanCOVIDVacc, cleanCCVI, cleanFoodInspection, cleanPopulation, \
    mergeData, splitTrainingData, importData

ccvi, COVstats, COVvacc, foodInsp, pop = importData()

def test_import_data():
    ccvi, COVstats, COVvacc, foodInsp, pop = importData()
    assert ccvi.size > 0
    assert COVstats.size > 0
    assert COVvacc.size > 0
    assert foodInsp.size > 0
    assert pop.size > 0

def test_cleanCOVIDStats():
    result = cleanCOVIDStats(COVstats)
    assert isinstance(result, pd.DataFrame)
    assert 'ZIP Code' in result.columns
    assert not result['ZIP Code'].isnull().any()
    assert not (result['ZIP Code'] == 'Unknown').any()
    assert result['ZIP Code'].dtype == 'int64'
    assert 'Cases - Weekly' in result.columns
    assert 'Deaths - Weekly' in result.columns
    assert 'Case Rate - Weekly' in result.columns
    assert result.groupby('ZIP Code').ngroups == result['ZIP Code'].nunique()

def test_cleanCOVIDVacc():
    result = cleanCOVIDVacc(COVvacc)
    assert isinstance(result, pd.DataFrame)
    assert 'Zip Code' in result.columns
    assert not result['Zip Code'].isnull().any()
    assert result['Zip Code'].dtype == 'int64'
    assert 'Total Doses - Daily' in result.columns
    assert result.groupby('Zip Code').ngroups == result['Zip Code'].nunique()

def test_cleanCCVI():
    result = cleanCCVI(ccvi)
    assert isinstance(result, pd.DataFrame)
    assert 'Community Area or ZIP Code' in result.columns
    assert not result['Community Area or ZIP Code'].isnull().any()
    assert 'CCVI Score' in result.columns

def test_cleanFoodInspection():
    result = cleanFoodInspection(foodInsp)
    assert isinstance(result, pd.DataFrame)
    assert 'Zip' in result.columns
    assert not result['Zip'].isnull().any()
    assert result['Zip'].dtype == 'int64'
    assert 'Results' in result.columns
    assert result['Results'].dtype == 'float64'
    assert (result['Results'] >= 0).all(), "Some values in 'Results' are less than 0"

def test_cleanPopulation():
    result = cleanPopulation(pop)
    assert isinstance(result, pd.DataFrame)
    assert 'Geography' in result.columns
    assert not result['Geography'].isnull().any()
    assert result['Geography'].dtype == 'int64'
    assert 'Population - Total' in result.columns
    assert not result.empty

def test_mergeData():
    COVstats_sample = pd.DataFrame({
        'ZIP Code': np.random.choice(range(10000, 99999), size=10, replace=False),
        'Cases - Weekly': np.random.randint(0, 500, size=10),
        'Case Rate - Weekly': np.random.uniform(0, 500, size=10),
        'Deaths - Weekly': np.random.randint(0, 20, size=10)
    })
    COVvacc_sample = pd.DataFrame({
        'Zip Code': np.random.choice(range(10000, 99999), size=10, replace=False),
        'Total Doses - Daily': np.random.randint(100, 1000, size=10)
    })
    ccvi_sample = pd.DataFrame({
        'Community Area or ZIP Code': np.random.choice(range(10000, 99999), size=10, replace=False),
        'CCVI Score': np.random.uniform(0, 1, size=10),
    })
    passFail_sample = pd.DataFrame({
        'Zip': np.random.choice(range(10000, 99999), size=10, replace=False),
        'Results': np.random.choice(range(1, 25), size=10)/5.0,
    })
    pop_sample = pd.DataFrame({
        'Geography': np.random.choice(range(10000, 99999), size=10, replace=False),
        'Population - Total': np.random.randint(1000, 100000, size=10)
    })
    actual_merged_data = mergeData(COVstats_sample, COVvacc_sample, pop_sample, passFail_sample, ccvi_sample)
    assert isinstance(actual_merged_data, pd.DataFrame)
    #Check unmodified columns
    assert 'CCVI Score' in actual_merged_data.columns, "CCVI Score column is missing after merge"
    assert 'ZIP Code' in actual_merged_data.columns, "ZIP Code column is missing after merge"
    #Check renamed columns
    assert 'Total COVID Cases' in actual_merged_data.columns, "Total COVID Cases column is missing after merge"
    assert 'Total COVID Deaths' in actual_merged_data.columns, "Total COVID Deaths column is missing after merge"
    assert 'Total COVID Vacc Doses' in actual_merged_data.columns, "Total COVID Vacc Doses column is missing after merge"
    assert 'Food Insp: Pass/Fail ratio' in actual_merged_data.columns, "Food Insp: Pass/Fail ratio column is missing after merge"
    # Assert each dropped column
    assert 'Zip Code' not in actual_merged_data.columns, "Zip Code column was not dropped"
    assert 'Zip' not in actual_merged_data.columns, "Zip column was not dropped"
    assert 'Community Area or ZIP Code' not in actual_merged_data.columns, "Community Area or ZIP Code column was not dropped"
    assert 'Location' not in actual_merged_data.columns, "Location column was not dropped"
    assert 'Geography' not in actual_merged_data.columns, "Geography column was not dropped"


# Test function using pytest
def test_splitTrainingData():
    merged_data = pd.DataFrame({
        'Total COVID Deaths': np.random.randint(0, 100, size=100),
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100)
    })
    X_train, X_test, y_train, y_test = splitTrainingData(merged_data)
    # Check if the split is correct (80% train, 20% test)
    assert len(X_train) == 80, "Incorrect training set size"
    assert len(X_test) == 20, "Incorrect testing set size"
    assert len(y_train) == 80, "Incorrect training set size for target"
    assert len(y_test) == 20, "Incorrect testing set size for target"
    # Check if the split is reproducible with the same random state
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(merged_data.drop('Total COVID Deaths', axis=1), merged_data['Total COVID Deaths'], test_size=0.2, random_state=42)
    assert X_train.equals(X_train_2), "Training sets are not equal"
    assert X_test.equals(X_test_2), "Testing sets are not equal"
    assert y_train.equals(y_train_2), "Training target sets are not equal"
    assert y_test.equals(y_test_2), "Testing target sets are not equal"

