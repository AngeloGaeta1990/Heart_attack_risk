import streamlit as st


def page_summary_body():
    """
    Function to print project summary summary
    """
    st.write("### Project Summary")
    st.info(
        "**Project Dataset**:\n"
        "The dataset represents anonymized patient data collected from five "
        "different hospitals: Budapest, Zurich, Basel, Long Beach and "
        "Cleveland.\n\n"

        "**Project Terms & Jargon**\n\n"
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

        "**HeartDisease**: represents if a patient has been affected by a "
        "myocardial infarction:\n"
        " - 1: Patient is affected\n\n"
        " - 0: Patient is not affected\n\n"
    )

    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file](https://github.com/AngeloGaeta1990/"
        "Heart_attack_risk/blob/main/README.md)."
    )

    st.success(
        "A physician provided anonymized patient data related to five "
        "different hospitals.\n\n"
        "The first business requirement is to understand which variables "
    )
