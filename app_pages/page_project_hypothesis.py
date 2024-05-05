import streamlit as st

def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    st.success(
        f"A physician provided anonymized patient data related to three different hospitals.\n\n"
        f" The physician would like a method to predict with at least 80% precision which patients are at risk of having a myocardial infarction.\n\n"
        f"Therefore: \n\n"
        f"The null hypothesis H0 is: Database features cannot predict myocardial infarction. \n\n"
        f" The alternative hypothesis H1 is: Database features can predict myocardial infarction.\n\n"
        f" This insight will be used by the physician to identify patients who are at risk of having a myocardial infarction."
    )
