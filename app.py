from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.page_correlation_study import page_correlation_study_body
from app_pages.page_heart_risk_analysis import page_hearth_risk_analysis_body
from app_pages.page_heart_risk_model_evaluation import (
     page_heart_risk_model_evaluation_body
)

app = MultiPage(app_name="Myocardial infarction risk")
app.add_page("Project Summary", page_summary_body)
app.add_page("Myocardial infarction correlation study",
             page_correlation_study_body)
app.add_page("Myocardial infarction risk analysis",
             page_hearth_risk_analysis_body)
app.add_page("Myocardial infarction model evalaution",
             page_heart_risk_model_evaluation_body)
app.run()
