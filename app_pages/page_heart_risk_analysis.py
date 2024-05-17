import streamlit as st
import pandas as pd
from src.data_management import load_heart_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_heart_risk


def page_hearth_risk_analysis_body():
    """
    function to run hearth risk analysis
    loads pipelines and inputs data
    writes legends for the data
    """

    version = 'v1'
    heart_risk_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_heart_disease/{version}/'
        f'pipeline_data_cleaning_feat_eng.pkl'
    )
    heart_risk_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/clf_pipeline.pkl"
    )
    heart_risk_features = (
        pd.read_csv(
            f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv"
        )
        .columns
        .to_list()
    )

    st.info(
        "The client is interested in determining whether it is possible to "
        "predict if a patient is at high risk of having a myocardial "
        "infarction.\n\n"
        "Based on the most relevant features, the practitioner decides which "
        "analysis to perform on each patient.\n\n"
        "Furthermore, patients at high risk of myocardial infarction will "
        "undergo specific therapy."
    )
    st.write("---")
    st.write("### Legend")
    st.info(
        "**Age**: age of the patient.\n\n"
        "**Sex**: **M** for male, **F** for female.\n\n"
        "**Chest Pain type**: describes the patient's chest pain and can be "
        "categorized as follows:\n"
        "- **Typical angina (TA)**: chest pain or discomfort caused by "
        "reduced oxygen-rich blood flow to the heart muscle.\n\n"
        "- **Atypical angina (ATA)**: chest pain or discomfort with symptoms "
        "not typical of angina pectoris.\n\n"
        "- **Non-anginal pain (NAP)**: chest pain or discomfort unrelated to "
        "angina or heart-related issues.\n\n"
        "- **Asymptomatic (ASY)**: a condition where a person does not "
        "exhibit any noticeable symptoms.\n\n"

        "**RestingBP**: refers to the measurement of blood pressure when a "
        "person is at rest, measured in mm Hg.\n\n"

        "**Cholesterol**: serum cholesterol level measured in mm/dl.\n\n"

        "**FastingBS**: blood sugar levels during fasting:\n\n"
        " - 1: FastingBS > 120 mg/dl\n\n"
        " - 0: FastingBS <= 120 mg/dl\n\n"

        "**RestingECG**: result of resting electrocardiogram (ECG), values "
        "are:\n\n"
        " - **Normal**\n\n"
        " - **ST**: represents the ST segment in an ECG.\n\n"
        " - **LVH**: left ventricular hypertrophy, indicating thickening or "
        "enlargement of the muscular wall of the left ventricle.\n\n"

        "**MaxHR**: maximum heart rate achieved in bpm.\n\n"

        "**ExerciseAngina**: describes if the patient is affected by angina "
        "pectoris after physical exercise (Y: yes, N: no).\n\n"

        "**Oldpeak**: measures the ST segment depression observed on an "
        "electrocardiogram (ECG) during exercise.\n\n"

        "**ST_slope**: indicates the slope of the ST segment during exercise, "
        "categorized as:\n"
        "  - **Up**: upsloping\n\n"
        "  - **Flat**: flat\n\n"
        "  - **Down**: downsloping\n\n"
    )

    X_live = DrawInputsWidgets()

    if st.button("Run Predictive Analysis"):
        heart_risk_prediction = predict_heart_risk(
            X_live, heart_risk_features, heart_risk_pipe_model,
            heart_risk_pipe_dc_fe
        )
        heart_risk_prediction


def DrawInputsWidgets():
    """
    function to add widgets
    """

    df = load_heart_data()
    cols = st.columns(3)
    X_live = pd.DataFrame([], index=[0])

    with cols[0]:
        feature = "Age"
        st_widget = st.number_input(
            label=feature,
            min_value=0,
            max_value=200,
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with cols[1]:
        feature = "Sex"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with cols[2]:
        feature = "ChestPainType"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with cols[0]:
        feature = "RestingBP"
        st_widget = st.number_input(
            label=feature,
            min_value=0,
            max_value=200,
            value=int(df[feature].median()),
            step=10
        )
    X_live[feature] = st_widget

    with cols[1]:
        feature = "Cholesterol"
        st_widget = st.number_input(
            label=feature,
            min_value=0,
            max_value=700,
            value=int(df[feature].median()),
            step=10
        )
    X_live[feature] = st_widget

    with cols[2]:
        feature = "FastingBS"
        st_widget = st.number_input(
            label=feature,
            min_value=0,
            max_value=1,
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with cols[0]:
        feature = "RestingECG"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with cols[1]:
        feature = "MaxHR"
        st_widget = st.number_input(
            label=feature,
            min_value=0,
            max_value=300,
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with cols[2]:
        feature = "ExerciseAngina"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with cols[0]:
        feature = "Oldpeak"
        st_widget = st.number_input(
            label=feature,
            min_value=-4,
            max_value=8,
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with cols[1]:
        feature = "ST_Slope"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget
    return X_live
