import streamlit as st

def page_summary_body():

    st.write("### Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(

        f"**Project Dataset**:\n"
        f"The dataset represents anonymized patient data collected from five different hospitals: "
        f"Budapest, Zurich, Basel, Long Beach and Cleveland.\n\n"

        f"**Project Terms & Jargon**\n\n"
        f"**Chest Pain type**: describes the patient's chest pain and can be categorized as follows:\n"
        f"- **Typical angina (TA)**: chest pain or discomfort caused by reduced oxygen-rich blood flow to the heart muscle.\n\n"
        f"- **Atypical angina (ATA)**: chest pain or discomfort with symptoms not typical of angina pectoris.\n\n"
        f"- **Non-anginal pain (NAP)**: chest pain or discomfort unrelated to angina or heart-related issues.\n\n"
        f"- **Asymptomatic (ASY)**: a condition where a person does not exhibit any noticeable symptoms.\n\n"

        f"**RestingBP**: refers to the measurement of blood pressure when a person is at rest, measured in mm Hg.\n\n"

        f"**Cholesterol**: serum cholesterol level measured in mm/dl.\n\n"

        f"**FastingBS**: blood sugar levels during fasting:\n\n"
        f" - 1: FastingBS > 120 mg/dl\n\n"
        f" - 0: FastingBS <= 120 mg/dl\n\n"

        f"**RestingECG**: result of resting electrocardiogram (ECG), values are:\n\n"
        f" - **Normal**\n\n"
        f" - **ST**: represents the ST segment in an ECG.\n\n"
        f" - **LVH**: left ventricular hypertrophy, indicating thickening or enlargement of the muscular wall of the left ventricle.\n\n"

        f"**MaxHR**: maximum heart rate achieved in bpm.\n\n"

        f"**ExerciseAngina**: describes if the patient is affected by angina pectoris after physical exercise (Y: yes, N: no).\n\n"

        f"**Oldpeak**: measures the ST segment depression observed on an electrocardiogram (ECG) during exercise.\n\n"

        f"**ST_slope**: indicates the slope of the ST segment during exercise, categorized as:\n"
        f"  - **Up**: upsloping\n\n"
        f"  - **Flat**: flat\n\n"
        f"  - **Down**: downsloping\n\n"

        f"**HeartDisease**: represents if a patient has been affected by a myocardial infarction:\n"
        f" - 1: Patient is affected\n\n"
        f" - 0: Patient is not affected\n\n"
    )

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/AngeloGaeta1990/Heart_attack_risk/blob/main/README.md).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"A physician provided anonymized patient data related to three different hospitals.\n\n"
        f"The first business requirement is to understand which variables correlate the most with a heart attack risk.\n\n "
        f"The second business requirement is to predict with at least 80% precision which patients are at risk of having a myocardial infarction, therefore:\n\n"
        f"The null hypothesis H0 is: Database features cannot predict myocardial infarction.\n\n"
        f"The alternative hypothesis H1 is: Database features can predict myocardial infarction.\n\n"
        f"This insight will be used by the physician to identify patients who are at risk of having a myocardial infarction."
        )
