import streamlit as st


def predict_heart_risk(X_live, heart_risk_features, pipeline_clf, pipeline_data_cleaning_feat_eng):

    # from live data, subset features related to this pipeline
    X_live_heart_risk = X_live.filter(heart_risk_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_heart_risk_dc_fe = pipeline_data_cleaning_feat_eng.transform(X_live_heart_risk)

    # predict
    heart_risk_prediction = pipeline_clf.predict(X_live_heart_risk_dc_fe)
    heart_risk_prediction_proba = pipeline_clf.predict_proba(X_live_heart_risk_dc_fe)
    

    # Create a logic to display the results
    heart_risk_prob = heart_risk_prediction_proba[0, heart_risk_prediction][0]*100
    if heart_risk_prediction == 1:
        heart_risk_result = 'High risk of myocardial infarction'
    else:
        heart_risk_result = 'Low risk of myocardial infarction'

    statement = (
        f'### There is {heart_risk_prob.round(1)}% probability '
        f'that this patient is at **{heart_risk_result}**.')

    st.write(statement)

    return heart_risk_prediction


# def predict_tenure(X_live, heart_risk_features, heart_risk_pipeline, heart_risk_labels_map):

#     # from live data, subset features related to this pipeline
#     X_live_heart_risk = X_live.filter(theart_risk_features)

#     # predict
#     heart_risk_prediction =  heart_risk_pipeline.predict(X_live_heart_risk)
#     heart_risk_prediction_proba = heart_risk_pipeline.predict_proba(X_live_heart_risk)
#     # st.write(tenure_prediction_proba)

#     # create a logic to display the results
#     proba = heart_risk_prediction_proba[0, heart_risk_prediction][0]*100
#     tenure_levels = tenure_labels_map[heart_risk_prediction[0]]

#     if tenure_prediction != 1:
#         statement = (
#             f"* In addition, there is a {proba.round(2)}% probability the prospect "
#             f"will stay **{tenure_levels} months**. "
#         )
#     else:
#         statement = (
#             f"* The model has predicted the prospect would stay **{tenure_levels} months**, "
#             f"however we acknowledge that the recall and precision levels for {tenure_levels} is not "
#             f"strong. The AI tends to identify potential churners, but for this prospect the AI is not "
#             f"confident enough on how long the prospect would stay."
#         )

#     st.write(statement)


