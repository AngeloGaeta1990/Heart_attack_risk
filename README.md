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

---
## Dataset

The dataset I used in this project is publicly available on [Kaggle](https://www.kaggle.com/) and can be downloaded and inspected here: [dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

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

 - As a user, I can get a measure of the myocardial infarction risk, 
 so that I can provide therapies to patient at high risk of myocardial infarction.

 - As a user, I can add patient data, so that I can get mycordial infarction risk for each patient.

 - As user, I can see which are the variables correlating the most with myocardial infarction risk, so that I can study if there is any biological correlation.

 - As a user, I can see how reliable is the ML model so that I can get an estimate of false posivites and false negatives.

- As a user, I can see which are the pipeline steps, so that I can have further insights on how the model was built.

-----
## ML model development





bug

xgboost size too large
parallel plot black line, becasue heartdisease NA
plarallel plot black line only if heart disease per leval was not run before
Pandas 2.20 not compatible with PPscore library
heroku port not found
ydata-profiling latest version not compatible with pandas < 2.20
best feature was not in the correct order for ROC curve plot




