### Myocardial infarction Risk Analysis
---
In an era marked by the increasing prevalence of cardiovascular diseases (CVDs), understanding the factors contributing to heart health and risk has become paramount. Cardiovascular diseases, encompassing conditions such as coronary artery disease, stroke, and hypertension, remain the leading cause of mortality worldwide. With lifestyle choices, genetic predispositions, and environmental factors interplaying in complex ways, the need for precise risk assessment models becomes evident.

This heart risk analysis project aims to delve into the intricate web of variables influencing cardiovascular health, employing advanced statistical techniques and machine learning algorithms. By analyzing vast datasets of anonymized patient data gathered from five different hospitals in Budapest, Zurich, Basel, Long Beach, and Cleveland, I aim to infer which variables from the dataset correlate most strongly with a high risk of myocardial infarction.

The primary objective is to provide a method capable of predicting, with at least 80% precision, which patients are at high risk. Subsequently, healthcare practitioners can proactively reach out to patients identified as high risk, offering them preventive therapies and interventions.

Live link to [Myocardial infarction Risk Analysis](https://heart-attack-risk-10ddd79e68a6.herokuapp.com/)
![Heart Risk Analysis](docs/images/live_website_page.png)

---
## Table of contents
---
- [Introduction](#heart-risk-analysis)
- [Dataset](#dataset)
- [Business requirements](#business-requirements)
- [Hypothesys](#Hypothesis)
- [User stories](#user-stories)
- [ML model development](#ml-model-development)
    - [Data Cleaning and Feature Engeenering](#data-cleaning-and-feature-engeenering)
    - [ML pipeline](#ml-pipeline)
        - [ML pipeline evaluation](#ml-pipeline-evaluation)
- [Correlation study](#correlation-study)

---
## Dataset

The dataset I used in this project is publicly available on [Kaggle](https://www.kaggle.com/) and can be downloaded and inspected here: [Heart failure dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

The dataset contains anonymized patient data collected from five different hospitals in Budapest, Zurich, Basel, Long Beach and Cleveland :
 - Cleveland: 303 observation
 - Budapest: 294 observations
 - Zurich: 123 observations
 - Long Beach: 200 observations
 - Basel: 270 observations

The dataset contains the following features:
- **Age**: Refers to the age of the patient

- **Sex**: Refers to the biological sex of the patient M for male, F for female

- **Chest Pain type**: Describes the patient's chest pain and can be categorized as follows:
        - **Typical angina (TA)**: Chest pain or discomfort caused by reduced oxygen-rich blood flow to the heart muscle.
        - **Atypical angina (ATA)**: Chest pain or discomfort with symptoms not typical of angina pectoris.
        - **Non-anginal pain (NAP)**: Chest pain or discomfort unrelated to angina or heart-related issues.
        - **Asymptomatic (ASY)**: A condition where a person does not exhibit any noticeable symptoms.

- **RestingBP**: Refers to the measurement of blood pressure when a person is at rest, measured in mm Hg.

- **Cholesterol**: Serum cholesterol level measured in mm/dl.

- **FastingBS**: Blood sugar levels during fasting:
    - 1: FastingBS > 120 mg/dl
    - 0: FastingBS <= 120 mg/dl

- **RestingECG**: Result of resting electrocardiogram (ECG), with values: 
    - **Normal**
    - **ST**: Represents the ST segment in an ECG.
    - **LVH**: Left ventricular hypertrophy, indicating thickening or enlargement of the muscular wall of the left ventricle.

- **MaxHR**: Maximum heart rate achieved in bpm.

- **ExerciseAngina**: Indicates if the patient is affected by angina pectoris after physical exercise (Y: yes, N: no

- **Oldpeak**: Measures the ST segment depression observed on an electrocardiogram (ECG) during exercise.

- **ST_slope**: Indicates the slope of the ST segment during exercise, categorized as:
    - **Up**: Upsloping
    - **Flat**: Flat
    - **Down**: Downsloping

The targer variable is :
- **HeartDisease**: Indicates if a patient is at high risk of myocardial infarction:
    - 1: High risk
    - 0: Low risk
 
---

## Business Requirements
The first business requirement is to identify which variables most strongly correlate with a high risk of myocardial infarction.

The second business requirement is to predict with at least 80% precision which patients are at high risk of having a myocardial infarction.

Phisicians will reach out to patients identified as at high risk and provide ad hoc therapies where necessary.

Key Stakeholders are : Physicians, Medical Practice, Hospitals and healthcare facilities.

Thus, the requirements are the following:
 - **Accessibility** : The tool developed will be used by physicians with no experience in coding; therefore, the interface must be accessible.

 - **Privacy**: Sensitive patient data must not be tracked, including names, last names, and email addresses.

- **Speed**: The physician should be able to add patient data and get results quickly.

- **Precision**: The method must have a precision of at least 80% to ensure that only patients at high risk of myocardial infarction are reached out to. Therefore, false positives and false negatives should be minimized.

----
## Hypothesis

In this project, the null hypothesis (H0) is the following: The database features cannot predict a high risk of myocardial infarction. The alternative hypothesis (H1) is: The database features can predict a high risk of myocardial infarction.

1. In order to predict if a patient is at high risk of myocardial infarction, I created an ML pipeline.

1. To evaluate the performance of the pipeline, the project includes a table of truth showing that precision is ≥ 80% on both the test and train sets.

1. I used Spearman, Pearson, and Predictive Power Score (PPS) to identify the variables that correlate the most. This allows for further research to determine if there is any biological reason to explain the correlations found.

1. Correlation and ML performance are shown through plots to make the performance and correlations clearly visible.

---

## User stories

Below, I added the user stories I used to build the project: 


 - As a user, I can get a measure of the myocardial infarction risk, 
 so that I can provide therapies to patient at high risk of myocardial infarction.

 - As a user, I can add patient data, so that I can get mycordial infarction risk for each patient.

 - As user, I can see which are the variables correlating the most with myocardial infarction risk, so that I can study if there is any biological correlation.

 - As a user, I can see how reliable is the ML model so that I can get an estimate of false posivites and false negatives.

- As a user, I can see which are the pipeline steps, so that I can have further insights on how the model was built.

-----
## ML model development

The ML model I implemented consists of two pipelines: one for data cleaning and feature engineering, and the second for feature selection and model building.

### Data Cleaning and Feature Engeenering

The data cleaning and feature engineering pipeline consists of the following steps: RandomSampleImputer, OrdinalCategoricalEncoder, and Winsorizer.


**RandomSampleInputer**. None of the features in the dataset had missing values. However, upon inspection, I noticed that approximately 20% of the values for the cholesterol feature were zeros.

![Raw data cholesterol histogram](/docs/images/raw_cholesterol_hist.png)

Therefore, I used the RandomSampleImputer to redistribute the data uniformly across the distribution. After applying the RandomSampleImputer, I obtained the following distribution:

![Random Sample inputer cholesterol histogram](/docs/images/random_sample_inputer_cholesterol.png)

**OridnalCategoricalEncoder**.  I used the OrdinalCategoricalEncoder to convert all the features of type object into numerical features. The object-type features are:
**Sex**, **ChestPainType**,**RestingECG**, **ExerciseAngina** and **ST_Slope**.


**Winsorizer**. I also noticed that the features **Cholesterol**, **Age**, **RestingBP**, and **Oldpeak** had outliers. Therefore, I applied the Winsorizer to all of them so that outliers are mapped to 1.5 times the interquartile range (IQR) of the values between the 25th and 75th percentiles.

![Cholesterol Winsorizer distribution](docs/images/cholesterol_winsorizer.png)

![Age Winsorizer distribution](docs/images/age_winsorizer.png)

![RestingBP Winsorizer distribution](docs/images/resting_bp_winsorizer.png)

![Oldpeak Winsorizer distribution](docs/images/oldpeak_winsorizer.png)

### ML pipeline

The ML pipeline instead consist of the following steps: **StandardScaler**, **FeatSelection**, **LogisticRegression**.

**StandardScaler**. It centers the data around the mean by subtracting the mean value of each feature from the data.

**FeatSelection**.  Fits the model to the data and selects the features based on their importance. This helps in reducing the dimensionality of the data and removing irrelevant features, which can improve model performance and reduce overfitting.

**LogisticRegression**. I used GridSearch to check which estimator would fit best with the dataset and got the following result:

![GridSearch reult](/docs/images/gridsearch_result.png)

Logistic Regression resulted in the estimator with the highest mean score. Therefore, I proceeded with hyperparameterization and model fitting. I divided the dataset into train (80% of the data) and test (20% of the data) sets.

#### ML pipeline evaluation

To evaluate the pipeline, I used the confusion matrix:

 - **Train Set**

    ![Train set table of truth](/docs/images/table_of_truth_train.png)


 - **Test Set**

    ![Test set table of truth](/docs/images/table_of_truth_test.png)

Furthermore, I also plotted the ROC curve to measure the difference between the method I implemented and a random sampler.

![ROC curve plot](/docs/images/roc_curve_plot.png)

Taken together, the results suggest that one of the two business requirements was addressed (see[Business requirements](#business-requirements)), as the precision is >80% on both the test and train sets. Additionally, the ROC curve highlighted a significant difference in the true and false positive rates compared to a random sampler.

---

## Correlation study

To study which features correlate the most with the risk of myocardial infarction, I used Spearman and Pearson tests, after converting all the object-type variables into numerical values with the OrdinalEncoder. Both tests ranked the features as follows:


| Rank | Feature          | Spearman Score | Pearson Score |
|------|------------------|----------------|---------------|
| 1    | ST_Slope         | 0.591913       | 0.558771      |
| 2    | ExerciseAngina   | 0.494282       | 0.494282      |
| 3    | ChestPainType    | 0.465971       | 0.459017      |
| 4    | Oldpeak          | 0.419046       | 0.403951      |
| 5    | MaxHR            | -0.404827      | -0.400421     |
| 6    | Sex              | -0.305445      | -0.305445     |
| 7    | Age              | 0.289576       | 0.282039      |
| 8    | FastingBS        | 0.267291       | 0.267291      |
| 9    | Cholesterol      | -0.139873      | -0.232741     |
| 10   | RestingBP        | 0.113860       | 0.107589      |


Both tests identified the same features as the ones that correlate the most with heart disease risk. Furthermore, I used a parallel plot to define the profile of a patient at high risk of myocardial infarction:

![Parallel plot](/docs/images/parallel_plot.png)

Here, we can see that a patient at high risk of myocardial infarction shows the following phenotype:

- **ST_Slope**. Flat or down 
- **ChestPainType**. Asymptomatic
- **ExerciseAngina**. The patient is affected by angina after physical exercise 
- **Oldpeak**. The old peak is > 0 
- **MAaxHR**. Has on average a lower heart rate

The above information can be considered by physicians to define the profile of patients at high risk of myocardial infarction.

---
bug

xgboost size too large
parallel plot black line, becasue heartdisease NA
plarallel plot black line only if heart disease per leval was not run before
Pandas 2.20 not compatible with PPscore library
heroku port not found
ydata-profiling latest version not compatible with pandas < 2.20
best feature was not in the correct order for ROC curve plot








