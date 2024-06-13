# !/usr/bin/env python
import pytest
import pandas as pd
import numpy as np
import mlflow
from src.scripts.MLOPSproject import cleanCOVIDStats, cleanCOVIDVacc, cleanCCVI, cleanFoodInspection, cleanPopulation, \
    mergeData, importData, nFold, linearReg, randomForestRegression, gbr, svr, calculateCrossValidationMetrics, \
    calculateMetrics

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


# Test function for nFold
def test_nFold(mock_mlflow):
    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [2, 3, 4, 5, 6],
        'Total COVID Deaths': [1, 1, 2, 2, 3]
    }
    mergedData = pd.DataFrame(data)
    lrMetrics, rfMetrics, gbrMetrics, svrMetrics = nFold(mergedData)
    assert len(lrMetrics) == 5, "Linear Regression metrics list should have 5 sets of metrics"
    assert len(rfMetrics) == 5, "Random Forest metrics list should have 5 sets of metrics"
    assert len(gbrMetrics) == 5, "Gradient Boosting metrics list should have 5 sets of metrics"
    assert len(svrMetrics) == 5, "Support Vector Regression metrics list should have 5 sets of metrics"


def test_calculateCrossValidationMetrics():
    performance_measures = [
        (0.5, 0.6, 0.7),
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6)
    ]
    mean, stdDev = calculateCrossValidationMetrics(performance_measures)
    expected_mean = np.array([0.3333, 0.4333, 0.5333])
    expected_stdDev = np.array([0.17, 0.17, 0.17])
    # Check if the calculated mean and stdDev are as expected
    assert np.allclose(mean, expected_mean, atol=0.0001), "The calculated mean is incorrect"
    assert np.allclose(stdDev, expected_stdDev, atol=0.0001), "The calculated standard deviation is incorrect"


# Test function for linearReg
def test_linearReg():
    np.random.seed(0)
    X_train = np.random.rand(10, 3)
    X_test = np.random.rand(10, 3)
    y_train = np.random.rand(10,)
    y_test = np.random.rand(10,)
    y_test_expected, y_pred_lr = linearReg(X_train, X_test, y_train, y_test)
    assert y_pred_lr.shape == y_test.shape, "The predicted values and test values do not match in shape."
    assert not np.any(np.isnan(y_pred_lr)), "The prediction contains NaN values."
    assert not np.any(np.isinf(y_pred_lr)), "The prediction contains infinite values."
    assert np.allclose(y_test, y_test_expected), "The expected test values and actual test values do not match."

# Unit test for randomForestRegression
def test_randomForestRegression():
        np.random.seed(0)
        X_train = np.random.rand(10, 3)
        X_test = np.random.rand(10, 3)
        y_train = np.random.rand(10, )
        y_test = np.random.rand(10, )
        y_test_expected, y_pred_rf = randomForestRegression(X_train, X_test, y_train, y_test)
        assert y_pred_rf.shape == y_test.shape, "The predicted values and test values do not match in shape."
        assert not np.any(np.isnan(y_pred_rf)), "The prediction contains NaN values."
        assert not np.any(np.isinf(y_pred_rf)), "The prediction contains infinite values."
        assert np.allclose(y_test, y_test_expected), "The expected test values and actual test values do not match."

# Unit test for gbr
def test_gbr():
        np.random.seed(0)
        X_train = np.random.rand(10, 3)
        X_test = np.random.rand(10, 3)
        y_train = np.random.rand(10, )
        y_test = np.random.rand(10, )
        y_test_expected, y_pred_gb = gbr(X_train, X_test, y_train, y_test)
        assert y_pred_gb.shape == y_test.shape, "The predicted values and test values do not match in shape."
        assert not np.any(np.isnan(y_pred_gb)), "The prediction contains NaN values."
        assert not np.any(np.isinf(y_pred_gb)), "The prediction contains infinite values."
        assert np.allclose(y_test, y_test_expected), "The expected test values and actual test values do not match."

# Unit test for svr
def test_svr():
        np.random.seed(0)
        X_train = np.random.rand(10, 3)
        X_test = np.random.rand(10, 3)
        y_train = np.random.rand(10, )
        y_test = np.random.rand(10, )
        y_test_expected, y_pred_svr = svr(X_train, X_test, y_train, y_test)
        assert y_pred_svr.shape == y_test.shape, "The predicted values and test values do not match in shape."
        assert not np.any(np.isnan(y_pred_svr)), "The prediction contains NaN values."
        assert not np.any(np.isinf(y_pred_svr)), "The prediction contains infinite values."
        assert np.allclose(y_test, y_test_expected), "The expected test values and actual test values do not match."

def test_calculateMetrics():
    # Test data
    y_test = np.array([1, 2, 3, 4, 5])
    y_pred_svr = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    # Expected results
    expected_mae_svr = np.mean(np.abs(y_test - y_pred_svr))
    expected_mse_svr = np.mean((y_test - y_pred_svr) ** 2)
    expected_rmse_svr = np.sqrt(expected_mse_svr)
    actual_mae_svr, actual_mse_svr, actual_rmse_svr = calculateMetrics(y_test, y_pred_svr)
    assert actual_mae_svr == pytest.approx(expected_mae_svr), "MAE does not match the expected value."
    assert actual_mse_svr == pytest.approx(expected_mse_svr), "MSE does not match the expected value."
    assert actual_rmse_svr == pytest.approx(expected_rmse_svr), "RMSE does not match the expected value."


# Mocking the MLflow to avoid actual logging during the test
class MockMlflow:
    def set_experiment(self, name):
        pass

    def start_run(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_metric(self, key, value):
        pass

@pytest.fixture
def mock_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow, "set_experiment", MockMlflow().set_experiment)
    monkeypatch.setattr(mlflow, "start_run", MockMlflow().start_run)
    monkeypatch.setattr(mlflow, "log_metric", MockMlflow().log_metric)
