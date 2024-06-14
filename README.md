# MLOPS-Project

Project created with MLOps-Template cookiecutter. For more info: https://mlopsstudygroup.github.io/mlops-guide/

## 📚 Proposal

Our project will use datasets from the Chicago Data Portal to forecast COVID-19 cases and deaths. We are planning on using 7 pieces of data for training and evaluation: Census economic data, Chicago Population Counts, Chicago COVID-19 Community Vulnerability Index (CCVI), Vaccinations by ZIP Code, Food Inspections, Public Health Statistics, and Cases, Tests, and Deaths by ZIP Code. Each of these datasets provides possible insights into whether or not a person is likely to catch COVID-19, and therefore having diverse data is important to make quality predictions. However, we could not find a single dataset that contained all of the data, so we will be using multiple datasets and combining them to create a single all-encompassing dataset that we will use for training and testing purposes. We will be using Scikit Learn and Pandas for most of this project because we are familiar with them, and they provide all the functionality we need. Firstly it enables us to preprocess the data by cleaning the datasets, handling missing values, removing duplicates, and standardizing data formats. This allows the data to be analyzed by an existing model, eliminates the noise from the data, and creates a clearer view of the data trends. We can also use these frameworks to create training and test data automatically from the full dataset, enabling us to train and test our model quickly and easily. Efficiency is extremely important in this case because we need to train and evaluate our model many times to find the ideal configuration. For example, we will aim to compare four different machine-learning algorithms on this dataset: Linear Regression, Random Forest Regression, Gradient Boosting Regression, and Support Vector Regression (SVR). Each of these algorithms has its benefits in certain use cases and we must test multiple to know which algorithm is the most effective for our dataset. By comparing the results using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) we can get a clear picture of which algorithm makes the best prediction for the given data.


## 📋 Requirements

* DVC
* Python3 and pip
* Access to IBM Cloud Object Storage

## 🏃🏻 Running Project

### 🔑 Setup IBM Bucket Credentials for IBM COS

#### MacOS and Linux
Setup your credentials on ```~/.aws/credentials``` and ```~/.aws/config```. DVC works perfectly with IBM Obejct Storage, although it uses S3 protocol, you can also see this in other portions of the repository.


~/.aws/credentials

```credentials
[default]
aws_access_key_id = {Key ID}
aws_secret_access_key = {Access Key}
```


### ✅ Pre-commit Testings

In order to activate pre-commit testing you need ```pre-commit```

Installing pre-commit with pip
```
pip install pre-commit
```

Installing pre-commit on your local repository. Keep in mind this creates a Github Hook.
```
pre-commit install
```

Now everytime you make a commit, it will run some tests defined on ```.pre-commit-config.yaml``` before allowing your commit.

**Example**
```
$ git commit -m "Example commit"

black....................................................................Passed
pytest-check.............................................................Passed
```


### ⚗️ Using DVC

Download data from the DVC repository(analog to ```git pull```)
```
dvc pull
```

Reproduces the pipeline using DVC
```
dvc repro
```
### Install Scikit-Learn
```
pip install scikit-learn
```

## Data Cleaning

Firstly we imported all of the datasets from CSV files into our program. For the COVID-19 stats, we only used the Zip code, weekly cases, weekly case rate, and weekly deaths. We then converted all data to int64 and grouped it by zip. We took the same steps for the COVID-19 vaccination data except that the only columns we used were zip code and doses per day, which we could use to find doses per week. For CCVI we only kept the Zip Code, CCVI Score and location info. The food inspection data required the most cleaning. Firstly, we had to convert the pass/fail results stored in the data into boolean values so they could be understood by the model. We then used a mask to only include entries that had a valid non-null boolean value. We then took only the Zip, Result and Location and grouped by Zip and Result. We then calculated the pass-to-fail ratio for each zip code in order to show which areas have a high amount of failed food inspections. For the public health stats, we used almost all of the data, only dropping Gonorrhea in both males and females. Finally for population data, we used a mask to filter out rows without geography data and used another mask to select data only from 2021, when COVID-19 was at its peak. Then, we created a mask that groups by geography(zip) and shows the total population of each. Our final step was to merge all of the data into one dataset using Panda’s merge function. We then removed the duplicate zip code columns and renamed a few of the columns to make the dataset easier to read, creating a single readable dataset to use for training and testing.

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
