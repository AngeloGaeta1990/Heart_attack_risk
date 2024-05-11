import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts

from app_pages.page_summary import page_summary_body
from app_pages.page_correlation_study import page_correlation_study_body
from app_pages.page_heart_risk_analysis import page_hearth_risk_analysis_body
from app_pages.page_heart_risk_model_evaluation import page_heart_risk_model_evaluation_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
# from app_pages.page_predict_churn import page_predict_churn_body
# from app_pages.page_predict_tenure import page_predict_tenure_body
# from app_pages.page_cluster import page_cluster_body

app = MultiPage(app_name= "Heart attack risk") 

# adding pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Myocardial infarction correlation study", page_correlation_study_body)
app.add_page("Myocardial infarction risk analysis",  page_hearth_risk_analysis_body)
app.add_page("Myocardial infarction model evalaution", page_heart_risk_model_evaluation_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
# app.add_page("ML: Prospect Churn", page_predict_churn_body)
# app.add_page("ML: Prospect Tenure", page_predict_tenure_body)
# app.add_page("ML: Cluster Analysis", page_cluster_body)

app.run() # Run the  app