import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_heart_data, load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_heart_risk_model_evaluation_body():

    # load heart risk pipeline files
    version = 'v1'
    heart_risk_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/clf_pipeline.pkl")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_test.csv")

    st.write("### ML Pipeline: Predict myocardial infarction")
    # display pipeline training summary conclusions
    st.info(
        f"Logistic regression proved to most suited algorithm for the study \n\n"
        f"It shows a precision > 80 on both train and test sets \n\n"
        f"0.384 and 0.285 of R2 Score on train and test sets respectively.\n\n "
        f"The feature selection component of the pipeline highlithed the the following as "
        f"The feature which correlates the most with myocardial infarction "
        f"**ST_Slope, ChestPainType, ExerciseAngina, Oldpeak, FastingBS, Sex** "
    )
    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict if a patient has an high risk of myocardial infarction")
    st.write(heart_risk_pipe)
    st.write("---")

    # show best features
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.write("---")

    # evaluate performance on both sets
    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=heart_risk_pipe,
                    label_map= ['Heart Disease', ' No Heart Disease'] )