import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_heart_data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_correlation_study_body():
    """
    Main function for page correlation study
    load dataset and perform distribution and parallel plot
    """

    df = load_heart_data()
    vars_to_study = ['ST_Slope', 'ChestPainType', 'ExerciseAngina',
                     'Oldpeak', 'MaxHR']

    st.write("### Myocardial infarction Study")
    st.info(
        "The client wants to understand which variables are most relevant "
        "and correlate with myocardial infarction."
    )

    if st.checkbox("Inspect dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows."
        )
        st.write(df.head(10))

    st.write("---")

    st.write(
        "A correlation study was conducted in the notebook to better "
        "understand how the variables are correlated with myocardial "
        "infarction risk. \n\n"
        "The study involved the use of Spearman and Pearson tests alongside "
        "the predictive power score(PPS).\n "
        "The test revealed that the most correlated variables are: "
        "**ST_Slope, ChestPainType, ExerciseAngina, Oldpeak, MaxHR**"
    )

    st.info(
        "The correlation indications and plots below converge in "
        "interpretation. They suggest that a patient at high risk of "
        "myocardial infarction exhibits the following characteristics: \n\n"
        "* Flat or down ST_slope \n"
        "* Asymptomatic chest pain \n"
        "* Has angina after exercise \n"
        "* Shows an old peak >4 \n"
        "* Has on average a lower maximum heart rate \n"
    )

    df_eda = df.filter(vars_to_study + ['HeartDisease'])
    df_eda = categorical_mapping(df_eda)

    if st.checkbox("Myocardial infarction risk per variable"):
        myocardial_risk_per_variable(df_eda)

    if st.checkbox("Parallel Plot"):
        st.write(
            "Information in dark blue indicates the profile of a patient "
            "at high risk of by myocardial infarction."
        )
        parallel_plot_heart_attack(df_eda)

        st.write(
            "A patient at high risk of myocardial infarction shows the "
            "following phenotype: \n\n"
            "- **ST_Slope**. Flat or down \n\n"
            "- **ChestPainType**. Asymptomatic \n\n"
            "- **ExerciseAngina**. The patient is affected by angina after "
            "physical exercise \n\n"
            "- **Oldpeak**. The old peak is > 0 \n\n"
            "- **MAaxHR**. Has on average a lower heart rate"
        )


def myocardial_risk_per_variable(df_eda):
    """
    function created using "Data exploration" notebook code -
    "Variables Distribution by Heart Attack risk" section
    """
    target_var = 'HeartDisease'
    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


def plot_categorical(df, col, target_var):
    """
    code copied from "Data exploration" notebook -
    "Variables Distribution by Heart Attack risk" section
    """
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var,
                  order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


def plot_numerical(df, col, target_var):
    """
    code copied from "Data exploration" notebook -
    "Variables Distribution by Heart Attack risk" section
    """
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


def parallel_plot_heart_attack(df):
    """
    function created using "Data exploration" notebook code -
    Parallel Plot section
    """
    vars_to_study = ['ST_Slope', 'ChestPainType', 'ExerciseAngina',
                     'Oldpeak', 'MaxHR', 'HeartDisease']
    df_eda = df[vars_to_study]
    df_eda['HeartDisease'] = df_eda['HeartDisease'].map(
        {'High Risk': 1, 'Low risk': 0})
    df_parallel = numerical_mapping(df_eda)
    fig = px.parallel_categories(df_parallel, color="HeartDisease")

    fig.update_layout({
        'font': {'size': 14, 'color': 'black', 'family': 'Arial, sans-serif'}
    })

    st.plotly_chart(fig)


def rename_bin_labels(column_name, map_name, disc):
    """
    function to rename bins in Heart rate column
    """
    n_classes = len(map_name) - 1
    classes_ranges = disc.binner_dict_[column_name][1:-1]

    labels_map = {}
    for n in range(0, n_classes):
        if n == 0:
            labels_map[n] = f"<{classes_ranges[0]}"
        elif n == n_classes-1:
            labels_map[n] = f"+{classes_ranges[-1]}"
        else:
            labels_map[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"
    return labels_map


def categorical_mapping(df):
    """
    function to value names in categorical variables
    """
    heart_disease_map = {1: 'High Risk', 0: 'Low risk'}
    chest_pain_map = {'ASY': 'Asymptomatic', 'NAP': 'Non Anginal Pain',
                      'ATA': 'Atypical Angina', 'TA': 'Typical Angina'}
    exercise_angina_map = {'N': 'No Angina', 'Y': 'Angina'}
    df['HeartDisease'] = df['HeartDisease'].map(heart_disease_map)
    df['ChestPainType'] = df['ChestPainType'].map(chest_pain_map)
    df['ExerciseAngina'] = df['ExerciseAngina'].map(exercise_angina_map)
    return df


def numerical_mapping(df):
    """
    function to replace numerical values with bins in oldpeak and maxhr
    variables
    """
    oldpeak_map = [-np.Inf, 0, 2, 4, np.Inf]
    max_hr_map = [-np.Inf, 100, 120, 140, 160, 180, np.Inf]
    disc = ArbitraryDiscretiser(
        binning_dict={'Oldpeak': oldpeak_map, 'MaxHR': max_hr_map})
    df_parallel = disc.fit_transform(df)
    max_hr_labels = rename_bin_labels('MaxHR', max_hr_map, disc)
    oldpeak_labels = rename_bin_labels('Oldpeak', oldpeak_map, disc)
    df_parallel['MaxHR'] = df_parallel['MaxHR'].replace(max_hr_labels)
    df_parallel['Oldpeak'] = df_parallel['Oldpeak'].replace(oldpeak_labels)
    return df_parallel
