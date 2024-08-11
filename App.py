import pandas as pd
import numpy as np
import joblib
import streamlit as st
from prophet import Prophet


# Load models and the label encoder
campaign_models = joblib.load('campaign_models.pkl')
label_encoder = joblib.load('label_encoder_new.pkl')

def get_forecast(future_dates, new_regressors, campaign_id, campaign_models):
    future = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
    future['Campaign_ID'] = campaign_id

    new_data = pd.DataFrame({
        'ds': pd.to_datetime(future_dates),
        'Campaign_ID': [campaign_id] * len(future_dates)
    })

    # Add regressors based on the input for each date
    for regressor, values in new_regressors.items():
        new_data[regressor] = values
        
    future = pd.merge(new_data, future, on=['ds', 'Campaign_ID'], how='inner')

    forecast = campaign_models[campaign_id].predict(future)
    predictions = forecast.loc[forecast['ds'].isin(future_dates), ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return predictions

# Streamlit app layout
st.title("Campaign Revenue Forecasting")

# Choose the Campaign ID from the list of keys in the models
campaign_id = st.selectbox("Select Campaign ID:", options=list(campaign_models.keys()))

# Input future dates using a date input that allows multiple dates
future_dates_input = st.date_input("Select Future Dates:", [], help="Select multiple dates for forecast", max_value=pd.to_datetime('2024-12-31'))

new_regressors = {
    'Audience_Reach': [],
    'Platform_Type_Encoded': []
}

# Check if dates have been selected
if future_dates_input:
    # Convert date input into a list of strings
    future_dates = [date.strftime('%Y-%m-%d') for date in future_dates_input]

    # Iterate over the selected future dates to create individual inputs
    for date in future_dates:
        st.subheader(f"Inputs for {date}")
        
        # Slider for Audience Reach for each date
        audience_reach = st.slider(f"Select Audience Reach for {date}:", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
        new_regressors['Audience_Reach'].append(audience_reach)
        
        # Provide original categorical options for Platform_Type
        platform_types = label_encoder.classes_  # This retrieves original categories
        selected_platform_type = st.selectbox(f"Select Platform Type for {date}:", options=platform_types)
        
        # Encode the platform type using the label encoder
        platform_encoded = label_encoder.transform([selected_platform_type])[0]
        new_regressors['Platform_Type_Encoded'].append(platform_encoded)

    if st.button("Get Predictions"):
        predictions = get_forecast(future_dates, new_regressors, campaign_id, campaign_models)
        # Rename the prediction columns
        predictions.columns = ['Date', 'Revenue', 'Minimum Revenue', 'Maximum Revenue']

        # Prepare HTML table for color-coded metrics
        for index, row in predictions.iterrows():
            st.subheader(f"Prediction for {row['Date'].date()}")
            table_html = f"""
            <table style='border-collapse: collapse; width: 100%;'>
                <tr>
                    <th style='text-align: left;'><span style='font-size: larger;'>Expected Revenue:</span><br>
                    <span style='color: blue; font-size: small;'>${row['Revenue']:.2f}</span></th>
                </tr>
                <tr>
                    <th style='text-align: left;'><span style='font-size: larger;'>Minimum Revenue:</span><br>
                    <span style='color: red; font-size: small;'>${row['Minimum Revenue']:.2f}</span></th>
                </tr>
                <tr>
                    <th style='text-align: left;'><span style='font-size: larger;'>Maximum Revenue:</span><br>
                    <span style='color: green; font-size: small;'>${row['Maximum Revenue']:.2f}</span></th>
                </tr>
            </table>
            """

            st.markdown(table_html, unsafe_allow_html=True)