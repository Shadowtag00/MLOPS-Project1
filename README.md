# MLOPS-Project

Project created with MLOps-Template cookiecutter. For more info: https://mlopsstudygroup.github.io/mlops-guide/

## 📚 Proposal

Our project will use datasets from the Chicago Data Portal to forecast COVID-19 cases and deaths. We are planning on using 7 pieces of data for training and evaluation: Census economic data, Chicago Population Counts, Chicago COVID-19 Community Vulnerability Index (CCVI), Vaccinations by ZIP Code, Food Inspections, Public Health Statistics, and Cases, Tests, and Deaths by ZIP Code. Each of these datasets provides possible insights into whether or not a person is likely to catch COVID-19, and therefore having diverse data is important to make quality predictions. However, we could not find a single dataset that contained all of the data, so we will be using multiple datasets and combining them to create a single all-encompassing dataset that we will use for training and testing purposes. We will be using Scikit Learn and Pandas for most of this project because we are familiar with them, and they provide all the functionality we need. Firstly it enables us to preprocess the data by cleaning the datasets, handling missing values, removing duplicates, and standardizing data formats. This allows the data to be analyzed by an existing model, eliminates the noise from the data, and creates a clearer view of the data trends. We can also use these frameworks to create training and test data automatically from the full dataset, enabling us to train and test our model quickly and easily. Efficiency is extremely important in this case because we need to train and evaluate our model many times to find the ideal configuration. For example, we will aim to compare four different machine-learning algorithms on this dataset: Linear Regression, Random Forest Regression, Gradient Boosting Regression, and Support Vector Regression (SVR). Each of these algorithms has its benefits in certain use cases and we must test multiple to know which algorithm is the most effective for our dataset. By comparing the results using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) we can get a clear picture of which algorithm makes the best prediction for the given data.


## 📋 Requirements

* Python3 and pip
* Docker
*pandas
*scikit-learn
*hydra-core
*prometheus_client
*rich
*mlflow
*dvc
*ruff
*pre-commit

## Installing and running docker
-Download and Install Docker Desktop for your platform of choice through the following link: https://docs.docker.com/engine/install/ .Then, in the project repository, open a terminal window and enter the following to start the program using a docker container:
```
docker build -t container .
docker run -it --rm container
```
We have the repo automatically building and pushing to docker hub on every push to main. This is the link to the image:
```
https://hub.docker.com/repository/docker/shadowtag00/mlops_final_project/
```

![Alt text](screenshots\docker_hub.png?raw=true "docker hub overview")

## Configuration Editing
### Hydra
Hydra is an open-source framework for configuring complex applications and workflows. It provides a powerful way to manage configuration parameters, compose configurations dynamically, and organize your codebase for flexibility and scalability. Hydra allows you to define and manage configuration parameters in a hierarchical and structured manner. You can organize configurations into groups, override parameters, and compose configurations from multiple sources. 
One big way we were able to use this tool was to have two separate configs for Profiling and Monitoring.

This run command will keep the program active so the user can take an extended look at metrics and monitoring:
```
docker run -p 8000:8000 -p 9090:9090 -p 5000:5000 -e CONFIG=monitoring -it --rm container
```
This run command will end the program immediately so that profiling statistics from cprofile can be shown:
```
docker run -it --rm -e CONFIG=profiling container
```
## Monitoring, Debugging and Experiment Tracking
### cProfiler
We are using cProfile for profiling the performance of the Python scripts we have written. To view and order the output, the following arguments were included in the Dockerfile:
```
-m cProfile -s pcalls
#if saving to file
-o data/cprofilerOutput.txt 
```
The Output will look as follows:
```
 ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      1    0.000    0.000   14.393   14.393 MLOPSproject.py:6(<module>)
      1    0.000    0.000   10.641   10.641 <decorator-gen-2>:1(main)
   10/1    0.003    0.000   10.641   10.641 context_managers.py:76(wrapped)
      1    0.000    0.000   10.641   10.641 main.py:80(decorated_main)
      1    0.000    0.000   10.638   10.638 utils.py:306(_run_hydra)
....
....
```
We sorted by the cumulative time and concluded that our program ran fairly efficiently aside from loading the data directly from the internet, which we could tell because the importData() function had a longer cumulative runtime than training and running the model itself. For this reason, we included the data in the directory and copied it directly to the container using the Dockerfile, eliminating the one obvious inefficiency in our code as the package would then execute in about 14 seconds and the individual methods were also relatively balanced in timing after fixing the data loading. 
### PDB
We used pdb, a debugger built into python, heavily at earlier stages of our project to create breakpoints where we suspected the data was not being processed correctly. We could then interact with the shell to see current variable values and analyze the state of the program just before or after it encountered an issue. This enabled us to quickly identify a typo when merging our data that caused one important dataset to be excluded from the calculations, which may have been missed otherwise.
### Logging
This program is designed to be run in two primary modes, INFO and DEBUG using rich profiling, automatically installed through docker. INFO includes basic information such as the model absolute error and squared error for each model and also notifies the user when each model is done running. DEBUG is much more step-by-step, and the program outputs the lengths of the data lists at various points to ensure it is being processed properly and logs different parts of the function, enabling us to identify what was going wrong when combining and processing our data. This allowed us to easily see where our data issues were occurring and address them directly without manually adding print statements that would need to be deleted. This is also extremely useful when working with multiple people and on a project being modified over time. A sample output is shown below:
```
DEBUG    pass_fail len:61
DEBUG    Population cleaned
....
INFO     Random Forest Regression Completed
INFO     Mean Absolute Error (MAE):39.505, Mean Squared Error (MSE):2354.3383833333332,Root Mean Squared Error (RMSE):48.52152494855591   
```
### Prometheus
Prometheus is an open-source monitoring and alerting system designed for reliability and scalability. It is widely used to collect metrics from various systems, allowing you to gain insights into the performance and health of your applications and infrastructure.
To open the webpage:
```
http://localhost:9090
```
Once at the webpage, query metrics using PromQL
For example: 
```
#Count of the models trained this session
model_training_count
```
We Integrated five metrics into the program to help us:
Time spent importing data (data_import_seconds)
Time spent cleaning data (data_cleaning_seconds)
Time spent training model (model_training_seconds)
Number of processed records (processed_records)
Number of times models are trained (model_training_count)

Being able to see these metrics over time and graphed with prometheus gives us a much better view into our program.
### MLFlow
MLflow allows you to track and compare experiments, logging parameters, metrics, and output files for each run. This helps you keep track of model performance and iterations.

To view metrics collected during the run, open:
```
http://localhost:5000
```
The MLFlow interface give a fantastic overview of all of the data and metrics of our program. All of the data is perfectly organized and easy to read. Having the abilty to organize the data by run and experiment helped us fine tune our models.
### Scikit-Learn
We use scikit-learn for the models in this assignment. It is automatically installed by docker or can be installed using the command below:
```
pip install scikit-learn
```

## Data Cleaning

Firstly we imported all of the datasets from CSV files into our program. For the COVID-19 stats, we only used the Zip code, weekly cases, weekly case rate, and weekly deaths. We then converted all data to int64 and grouped it by zip. We took the same steps for the COVID-19 vaccination data except that the only columns we used were zip code and doses per day, which we could use to find doses per week. For CCVI we only kept the Zip Code, CCVI Score and location info. The food inspection data required the most cleaning. Firstly, we had to convert the pass/fail results stored in the data into boolean values so they could be understood by the model. We then used a mask to only include entries that had a valid non-null boolean value. We then took only the Zip, Result and Location and grouped by Zip and Result. We then calculated the pass-to-fail ratio for each zip code in order to show which areas have a high amount of failed food inspections. For the public health stats, we used almost all of the data, only dropping Gonorrhea in both males and females. Finally for population data, we used a mask to filter out rows without geography data and used another mask to select data only from 2021, when COVID-19 was at its peak. Then, we created a mask that groups by geography(zip) and shows the total population of each. Our final step was to merge all of the data into one dataset using Panda’s merge function. We then removed the duplicate zip code columns and renamed a few of the columns to make the dataset easier to read, creating a single readable dataset to use for training and testing.

From our testing, we found that Linear Regression, Random Forest Regression, Gradient Boosting Regression, and Support Vector Regression (SVR) were reasonably effective, but overall Random Forest regression delivered the best results for our use case. We concluded this by analyzing the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). We also used cross-validation techniques, and the results supported our original findings. Random Forest Regression had the lowest MAE and RMSE values in our tests, leading us to our conclusion.

## Pytest Unit Tests
We created unit tests to thoroughly test the data processing, model training, and evaluation scripts to ensure all error-prone aspects of our program are bug free. In practice, we used pytest to ensure the correct columns were being added and removed, that data was the right size at various parts of the data and also tested each model by training them and seeing if their guess was within a small enough margin to be useful. We provide each function with dummy data in order to quickly and extensively test each function. To run all tests, call the following from the project's root directory:
```
python src/tests/test_MLOPS-Project.py
```
Sample output:
```
============================= test session starts =============================
collecting ... collected 14 items

test_MLOPS-Project.py::test_import_data PASSED                           [  7%]
test_MLOPS-Project.py::test_cleanCOVIDStats PASSED                       [ 14%]
test_MLOPS-Project.py::test_cleanCOVIDVacc PASSED                        [ 21%]
test_MLOPS-Project.py::test_cleanCCVI PASSED                             [ 28%]
test_MLOPS-Project.py::test_cleanFoodInspection PASSED                   [ 35%]
test_MLOPS-Project.py::test_cleanPopulation PASSED                       [ 42%]
test_MLOPS-Project.py::test_mergeData PASSED                             [ 50%]
test_MLOPS-Project.py::test_nFold PASSED                                 [ 57%]
test_MLOPS-Project.py::test_calculateCrossValidationMetrics PASSED       [ 64%]
test_MLOPS-Project.py::test_linearReg PASSED                             [ 71%]
test_MLOPS-Project.py::test_randomForestRegression PASSED                [ 78%]
test_MLOPS-Project.py::test_gbr PASSED                                   [ 85%]
test_MLOPS-Project.py::test_svr PASSED                                   [ 92%]
test_MLOPS-Project.py::test_calculateMetrics PASSED                      [100%]

```
## DVC
We used DVC to control versioning and storing data from data models in an outside server, such as Google Drive. This made it much easier to run our model quickly and efficiently without needing to manually push data changes to git every time. It also enables reproducibility and traceability as it documents the data fed to the model at each step. DVC is automatically run when using the dockerfile to run the project

## CML and Contious Docker Building
We leveraged CML and our docker file in order to implement integration. GitHub actions/pre-commit statements allowed us to run and test our repository automatically in the GitHub server to verify it is running as intended. This will automatically be run on each commit and push, no extra steps are required.

## GCP and Deployment
### GCP Artifact Registry
GCP Artifact Registry involves managing and storing Docker images securely. This includes: Setting up the Artifact Registry: Configure the registry in GCP and granting necessary permissions, creating a Docker Image using the provided Dockerfile and pushing to the registry by Authenticate with GCP and pushing the built image to the Artifact Registry.
### Custom Training Job
This involved preparing the GCP environment for training, running the job itself from a bucket, and storing the results via DVC
### FastAPI
Fast API allows us to create an easy way to access functionality from our program quickly and easily in other applications. This would enable other applications to easily implement and call our model without storing locally
### Dockerization and Deployment
Finally, we Containerized our model with the Dockerfile. We then deployed the container to Cloud Run so our model is easy to distribute and for others to access. Lastly, we tested that it ran as intended in the cloud after deployment.
## Results
From our testing, we found that Linear Regression, Random Forest Regression, Gradient Boosting Regression, and Support Vector Regression (SVR) were reasonably effective, but overall Random Forest regression delivered the best results for our use case. We concluded this by analyzing the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). We also used cross-validation techniques to ensure the performance and reliability of the models. Random Forest Regression had the lowest MAE and RMSE values in our tests, leading us to our conclusion.

## Challenges and Improvements

One unexpected challenge we faced was finding the necessary data for our chosen topic. There was no single data source that provided all of the data we were looking for so we spent a lot of time looking for data and combining data sources into one table. This process was more complicated than anticipated because each table had to be grouped by zip, and each had to be processed separately before it could be added to the larger dataset. 

One place where we could’ve improved is by testing more different layer parameters. We focused much of our testing on finding the best model, and in future weeks I would like to test the other parameters more thoroughly to improve our code’s effectiveness. The other major change I would make is using a simpler dataset. In this section, we spent a lot of time processing and combining data when we could’ve focused our energy elsewhere, We have mostly moved past this issue, but it is something we will remember in the future.

