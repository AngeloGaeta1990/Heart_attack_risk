import streamlit as st


def predict_heart_risk(X_live, heart_risk_features, pipeline_clf,
                       pipeline_data_cleaning_feat_eng):

    """
    From live data, subset features related to this pipeline
    Applies data cleaning / feat engine pipeline to live data
    Creates a logic to display the results
    """
    X_live_heart_risk = X_live.filter(heart_risk_features)
    X_live_heart_risk_dc_fe = pipeline_data_cleaning_feat_eng.transform(
        X_live_heart_risk)
    heart_risk_prediction = pipeline_clf.predict(X_live_heart_risk_dc_fe)
    heart_risk_prediction_proba = pipeline_clf.predict_proba(
        X_live_heart_risk_dc_fe)
    heart_risk_prob = heart_risk_prediction_proba[
        0, heart_risk_prediction][0]*100
    if heart_risk_prediction == 1:
        heart_risk_result = 'High risk of myocardial infarction'
    else:
        heart_risk_result = 'Low risk of myocardial infarction'

    statement = (
        f'### There is {heart_risk_prob.round(1)}% probability '
        f'that this patient is at **{heart_risk_result}**.')

    st.write(statement)
    return heart_risk_prediction
