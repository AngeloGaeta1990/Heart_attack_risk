import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc
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

    st.write('---')
    st.write('### Error score')
    regression_performance(X_train, y_train, X_test, y_test, heart_risk_pipe)

    if st.checkbox("ROC curve analysis"):
        classifier_instance = heart_risk_pipe.named_steps['model']
        best_features =['ST_Slope', 'ChestPainType', 'ExerciseAngina', 'Oldpeak', 'FastingBS', 'Sex']
        plot_roc_curve_classifier(classifier_instance, X_test, y_test, best_features)
        



def plot_roc_curve_classifier(classifier, X_test, y_test, best_features):
    # Select only the relevant columns from the test data
    X_test_selected = X_test[best_features]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test_selected)[:, 1])
    
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess'))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0, y=1, orientation='h'),
        margin=dict(l=50, r=50, t=50, b=50),
        width=800,
        height=600
    )
    st.plotly_chart(fig)

def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    st.write('R2 Score:', r2_score(y, prediction).round(3))
    st.write('Mean Absolute Error:', mean_absolute_error(y, prediction).round(3))
    st.write('Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    st.write('Root Mean Squared Error:', np.sqrt(
        mean_squared_error(y, prediction)).round(3))


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("**Model Evaluation** \n")
    st.write("**Train Set**")
    regression_evaluation(X_train, y_train, pipeline)
    st.write("**Test Set**")
    regression_evaluation(X_test, y_test, pipeline)