import streamlit as st
import pandas as pd
from src.data_management import load_heart_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_heart_risk


def page_hearth_risk_analysis_body():

    # load predict heart risk files
    version = 'v1'
    heart_risk_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_heart_disease/{version}/pipeline_data_cleaning_feat_eng.pkl')
    heart_risk_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/clf_pipeline.pkl")
    heart_risk_features = (pd.read_csv(f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv")
                      .columns
                      .to_list()
                      )

    st.write("### Prospect Churnometer Interface")
    st.info(
        f"* The client is interested in determining whether it possible to predict if a patient is "
        f"at high risk of having a myocordial infarction \n\n"
        f"Based on the most relevant features the practicioner decides which analysis perform on each patient \n\n "
        f"Furthermore patients at high risk of myocardial infarction will undergo to a specific therapy "
    )
    st.write("---")

    # Generate Live Data
    # check_variables_for_UI(tenure_features, churn_features, cluster_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        heart_risk_prediction = predict_heart_risk(
            X_live, heart_risk_features, heart_risk_pipe_model, heart_risk_pipe_dc_fe)

def check_variables_for_UI(heart_risk_features):
    import itertools

    # The widgets inputs are the features used in all pipelines (tenure, churn, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(heart_risk_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # load dataset
    df =  load_heart_data()

# we create input widgets only for 6 features
    # col2, col3, col6 = st.beta_columns(4)
    # col9, col10, col11 = st.beta_columns(4)
    cols = st.columns(3)

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    
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

    # st.write(X_live)

    return X_live