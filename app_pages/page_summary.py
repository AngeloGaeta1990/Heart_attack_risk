import streamlit as st


def page_summary_body():
    """
    Function to print project summary summary
    """
    st.write("### Project Summary")
    st.info(
        "**Project Dataset**:\n"
        "The dataset represents anonymized patient data collected from five "
        "different hospitals in Budapest, Zurich, Basel, Long Beach and "
        "Cleveland.\n\n"

        "**Project Terms & Jargon**\n\n"
        "**Chest Pain type**: Describes the patient's chest pain and can be "
        "categorized as follows:\n"
        "- **Typical angina (TA)**: Chest pain or discomfort caused by "
        "reduced oxygen-rich blood flow to the heart muscle.\n\n"
        "- **Atypical angina (ATA)**: Chest pain or discomfort with symptoms "
        "not typical of angina pectoris.\n\n"
        "- **Non-anginal pain (NAP)**: Chest pain or discomfort unrelated to "
        "angina or heart-related issues.\n\n"
        "- **Asymptomatic (ASY)**: A condition where a person does not "
        "exhibit any noticeable symptoms.\n\n"

        "**RestingBP**: Refers to the measurement of blood pressure when a "
        "person is at rest, measured in mm Hg.\n\n"

        "**Cholesterol**: Serum cholesterol level measured in mm/dl.\n\n"

        "**FastingBS**: Blood sugar levels during fasting:\n\n"
        " - 1: FastingBS > 120 mg/dl\n\n"
        " - 0: FastingBS <= 120 mg/dl\n\n"

        "**RestingECG**: Result of resting electrocardiogram (ECG), with "
        "values: \n\n"
        " - **Normal**\n\n"
        " - **ST**: Represents the ST segment in an ECG.\n\n"
        " - **LVH**: Left ventricular hypertrophy, indicating thickening or "
        "enlargement of the muscular wall of the left ventricle.\n\n"

        "**MaxHR**: Maximum heart rate achieved in bpm.\n\n"

        "**ExerciseAngina**: Indicates if the patient is affected by angina "
        "pectoris after physical exercise (Y: yes, N: no).\n\n"

        "**Oldpeak**: Measures the ST segment depression observed on an "
        "electrocardiogram (ECG) during exercise.\n\n"

        "**ST_slope**: Indicates the slope of the ST segment during exercise, "
        "categorized as:\n"
        "  - **Up**: Upsloping\n\n"
        "  - **Flat**: Flat\n\n"
        "  - **Down**: Downsloping\n\n"

        "**HeartDisease**: Indicates if a patient is at high risk of "
        "myocardial infarction:\n"
        " - 1: High risk\n\n"
        " - 0: Low risk\n\n"
    )

    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file](https://github.com/AngeloGaeta1990/"
        "Heart_attack_risk/blob/main/README.md)."
    )

    st.write("### Project Aim")
    st.success(
        "A physician provided anonymized patient data from five "
        "different hospitals.\n\n"
        "The first business requirement is to identify which variables "
        "most strongly correlate with a high risk of myoardial infarction \n\n"
        "The second business requirement is to predict with at least 80% "
        "precision which patients are at high risk of having a myocardial "
        "infarction. Therefore:\n\n"
        "The **null hypothesis(H0)** is: The database features cannot predict "
        "a high risk of myocardial infarction.\n\n"
        "The **alternative hypothesis(H1)**  is: The database features can "
        "predict a high risk of myocardial infarction.\n\n"
        "These insights will be used by physicians to identify patients "
        " at high risk of myocardial infarction and to provide "
        "preventive therapies"
    )
