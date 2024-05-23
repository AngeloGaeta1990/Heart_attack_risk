import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, roc_curve, auc)
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_heart_risk_model_evaluation_body():
    """
    function to load heart risk pipeline files
    show pipeline steps
    show best features plots
    show table of truth
    show roc curve
    """
    version = 'v1'
    heart_risk_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/clf_pipeline.pkl"
        )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_test.csv")

    st.write("### ML Pipeline: Predict myocardial infarction")
    st.info(
       "Logistic regression proved to be the most suited algorithm"
       "for the study. \n\n"
       "It shows a precision greater than 80% on both train and test "
       "sets.\n\n"
       "0.384 and 0.285 of R2 Score on train and test sets respectively.\n\n"
       "The feature selection component of the pipeline highlighted the "
       "following as "
       "the features which correlate the most with myocardial infarction "
       "risk:\n\n"
       "**ST_Slope, ChestPainType, ExerciseAngina, Oldpeak, FastingBS, Sex**."
    )
    st.write("---")

    st.write("* ML pipeline to predict if a patient has an high risk of "
             "myocardial infarction")
    st.write(heart_risk_pipe)
    st.write("---")

    st.write("* The features the model was trained on")
    st.write(X_train.columns.to_list())
    st.write("---")
    if st.checkbox("Feature Importance"):
        feature_importance_plot(heart_risk_pipe, X_train)
        st.info("Shows the most important feature selected by the feat"
                "selection step of the pipeline, ranked by absolute "
                "coefficient.\n\n"
                "**1.ChestPainType**\n\n"
                "**2.ST_slope** \n\n"
                "**3.Oldpeak** \n\n"
                "**4.FastingBS** \n\n"
                "**5.ExerciseAngina**\n\n"
                "**6.Sex**\n\n")
        st.write("")

    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=heart_risk_pipe,
                    label_map=['Heart Disease', ' No Heart Disease'])

    st.write('---')
    st.write('### Error score')
    regression_performance(X_train, y_train, X_test, y_test, heart_risk_pipe)

    if st.checkbox("ROC curve analysis"):
        best_features = ['Sex', 'ChestPainType', 'FastingBS', 'ExerciseAngina',
                         'Oldpeak', 'ST_Slope']
        plot_roc_curve_classifier(heart_risk_pipe, X_test, y_test,
                                  best_features)
        st.info("The ROC curve compares the model's results to those of a "
                "random sampler, and plots the sensitivity against "
                "1-specifity.\n\n"
                "The AUC of 0.88 proves the excellent ability of the "
                "model to distinguish between low-risk and high-risk patients")


def plot_roc_curve_classifier(pipeline, X_test, y_test, best_features):
    """
    function to plot roc curve
    """
    classifier = pipeline.named_steps['model']
    X_test_selected = X_test[best_features]
    fpr, tpr, thresholds = roc_curve(y_test,
                                     classifier.predict_proba(X_test_selected)
                                     [:, 1])
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name='ROC curve (area = %0.2f)' % roc_auc))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                  line=dict(dash='dash'), name='Random Guess'))
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
    """
    function to evaluate error score
    """
    prediction = pipeline.predict(X)
    st.write('R2 Score:', r2_score(y, prediction).round(3))
    st.write('Mean Absolute Error:',
             mean_absolute_error(y, prediction).round(3))
    st.write('Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    st.write('Root Mean Squared Error:', np.sqrt(
        mean_squared_error(y, prediction)).round(3))


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    function to evaluate error score in Train and Test sets
    """
    st.write("**Model Evaluation** \n")
    st.write("**Train Set**")
    regression_evaluation(X_train, y_train, pipeline)
    st.write("**Test Set**")
    regression_evaluation(X_test, y_test, pipeline)


def feature_importance_plot(pipeline_clf, X_train):
    """
    function to plot the feature importance
    """
    coefficients = pipeline_clf['model'].coef_[0]
    best_features = X_train.columns[
        pipeline_clf['feat_selection'].get_support()
        ]
    df_coefficients = pd.DataFrame({'Feature': best_features,
                                    'Coefficient': coefficients})
    df_coefficients_sorted = df_coefficients.reindex(
      df_coefficients['Coefficient'].abs().sort_values(ascending=False).index)

    fig = go.Figure(go.Bar(
        x=df_coefficients_sorted['Coefficient'],
        y=df_coefficients_sorted['Feature'],
        orientation='h',
        marker=dict(color='skyblue')
    ))

    fig.update_layout(
        title='Feature Importance (Absolute Coefficients)',
        xaxis_title='Coefficient Magnitude',
        yaxis_title='Feature',
        yaxis=dict(tickmode='linear'),
        margin=dict(l=100, r=20, t=50, b=50),
        width=800,
        height=600
    )
    st.plotly_chart(fig)
