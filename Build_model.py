import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, chi2
import seaborn as sns
import joblib
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, norm, chi2
import statsmodels.tsa.stattools as ts
color_pal = sns.color_palette()
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os
import mysql.connector
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


# Establish connection to MySQL database
load_dotenv()

cnx = mysql.connector.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    passwd=os.getenv('DB_PASS'),
    db=os.getenv('DB_NAME')
)

# Create cursor object
c = cnx.cursor()


def view_all_data():
    c.execute('SELECT * FROM dashboard_analytics.combined_data;')
    data = c.fetchall()
    return data


data = view_all_data()
df = pd.DataFrame(data, columns=['Campaign_ID', 'Date_Time', 'Platform_Type', 'Impressions', 'Clicks', 'Conversions', 'Tracked_Ads', 'Cost', 'Revenue', 'Average_Frequency', 'Audience_Reach', 'Unique_Reach', 'On_Target_Impressions', 'Audience_Efficiency_Rate', 'Percentage_On_Target'])
df.set_index('Date_Time', inplace=True)
## Convert 'object' type features to numeric, except for 'Campaign_ID'

def convert_to_numeric(df):
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        if col != 'Date_Time' and col != 'Campaign_ID' and col != 'Platform_Type':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                print(f"Unable to convert '{col}' to numeric. Skipping.")
        elif col == 'Campaign_ID' and col == 'Platform_Type':
            print(f"Skipping conversion of 'Campaign_ID' and ' Platform_Type' columns.")
        elif col == 'Date_Time':
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

df = convert_to_numeric(df)


## Create a copy of data
data1 = df.copy()


data1_reset = data1.reset_index()

# Assuming 'data1_reset' is your DataFrame that includes 'Campaign_ID'
cleaned_data = pd.DataFrame()  
unique_campaigns = data1_reset['Campaign_ID'].unique()

for campaign in unique_campaigns:
    # Filter the DataFrame for the current campaign
    feature_campaign_data = data1_reset[data1_reset['Campaign_ID'] == campaign]

    print(f"Processing Campaign ID: {campaign}")
    
    # Start with campaign data as cleaned data
    cleaned_campaign_data = feature_campaign_data.copy()
    
    for column in feature_campaign_data.columns:
        feature_data = feature_campaign_data[column]
        
        if feature_data.dtype in ['float64', 'int64']:
            # Z-score method
            z_scores = (feature_data - feature_data.mean()) / feature_data.std()
            outliers_z = np.abs(z_scores) > 3

            # Tukey's method
            q1 = feature_data.quantile(0.25)
            q3 = feature_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_tukey = (feature_data < lower_bound) | (feature_data > upper_bound)

            # Combined outliers
            combined_outliers = outliers_z | outliers_tukey
            
            # If outliers are found, mark outliers_found and remove them
            if combined_outliers.any():
                outliers_found = True
                cleaned_campaign_data = cleaned_campaign_data[~combined_outliers]
                print(f"  Feature: {column} - Outliers removed (combined): {combined_outliers.sum()} out of {len(feature_data)}")


    if not outliers_found:
        break  # Exit loop if no outliers found

    # Drop duplicates from cleaned campaign data if any
    cleaned_campaign_data = cleaned_campaign_data.drop_duplicates()

    # Append cleaned campaign data to the cleaned_data DataFrame
    cleaned_data = pd.concat([cleaned_data, cleaned_campaign_data], ignore_index=True)

# Verification of cleaned_data
for column in cleaned_data.columns:
    feature_dataw = cleaned_data[column]
    if feature_dataw.dtype in ['float64', 'int64']:
        z_scores = (feature_dataw - feature_dataw.mean()) / feature_dataw.std()
        outliers_z = np.where(np.abs(z_scores) > 3)

        q1 = feature_dataw.quantile(0.25)
        q3 = feature_dataw.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_tukey = np.where((feature_dataw < lower_bound) | (feature_dataw > upper_bound))

        print(f"Feature: {column}")
        print(f"Outliers after cleaning (Z-score): {len(outliers_z[0])} out of {len(feature_dataw)}")
        print(f"Outliers after cleaning (Tukey's method): {len(outliers_tukey[0])} out of {len(feature_dataw)}")

#One-Hot encoding for Platform_Type feature
label_encoder = LabelEncoder()
# Fit and transform the column
encoded_data = label_encoder.fit_transform(cleaned_data['Platform_Type'])
# Create a new DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=['Platform_Type_Encoded'])

# Concatenate the encoded DataFrame with the copied DataFrame
data2 = pd.concat([cleaned_data, encoded_df], axis=1)

data2 = data2.rename(columns={'Date_Time': 'ds', 'Revenue': 'y'})


# Ensure that 'ds' is of datetime type
data2['ds'] = pd.to_datetime(data2['ds'])

# Define the date range as datetime objects
train_start = pd.to_datetime('2024-03-01')
train_end = pd.to_datetime('2024-05-31')
test_start = pd.to_datetime('2024-06-01')
test_end = pd.to_datetime('2024-06-30')

# Split the data into train and validation sets
train = data2.loc[data2['ds'] < train_end]
valid = data2.loc[(data2['ds'] >= train_end) & (data2['ds'] < test_end)]

# Training Prophet

wmape_results = []
p = list()
campaign_models = {}
for campaign in train['Campaign_ID'].unique():
    print('Campaign:', campaign)
    train_ = train.loc[train['Campaign_ID'] == campaign]
    valid_ = valid.loc[valid['Campaign_ID'] == campaign]
    
    m = Prophet(seasonality_mode='additive', 
                weekly_seasonality=True,
                daily_seasonality=True,
                )
    # Add regressors
    regressors = ['Audience_Reach',
                  'Platform_Type_Encoded']
    
    for regressor in regressors:
        m.add_regressor(regressor)

    m.fit(train_)
    campaign_models[campaign] = m
    # Save campaign models
    joblib.dump(campaign_models, 'campaign_models.pkl')

    # Save Label Encoder for Platform_Type
    joblib.dump(label_encoder, 'label_encoder_new.pkl')

    future = m.make_future_dataframe(periods=30, include_history=False)
    future = future.merge(valid_[['ds'] + regressors ], on='ds', how='left')
    forecast = m.predict(future)
    forecast['Campaign_ID'] = campaign
    p.append(forecast[['ds', 'yhat', 'Campaign_ID']])
    # Calculate WMAPE for the current campaign
    forecast = forecast.merge(valid_[['ds', 'y']], on='ds', how='left')  # Merge to get actual values
    wmape_val = wmape(forecast['y'], forecast['yhat'])

    # Store the WMAPE result for the campaign
    wmape_results.append({'Campaign_ID': campaign, 'WMAPE': wmape_val})
    
p = pd.concat(p, ignore_index=True)
p['yhat'] = p['yhat'].clip(lower=0)
p = p.merge(valid, on=['ds', 'Campaign_ID'], how='left')
wmape(p['y'], p['yhat'])

# Convert WMAPE results to DataFrame
wmape_df = pd.DataFrame(wmape_results)

# Display or save the WMAPE results per campaign
print(wmape_df)


# Predict future data


def get_forecast(future_dates, new_regressors, campaign_id, campaign_models):
    """
    Forecast values based on future dates and provided regressors.
    
    Parameters:
    - future_dates: pd.Series or list, the future dates for which to predict values.
    - new_regressors: dict, regressor values corresponding to future dates.
    - campaign_id: str, unique identifier for the campaign.
    - campaign_models: dict, mapping of campaign IDs to trained models.

    Returns:
    - predictions: DataFrame, the predicted target values for all forecast dates.
    """
    # Create a future DataFrame with future_dates
    future = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
    future['Campaign_ID'] = campaign_id

    # Combine new regressors with future DataFrame
    new_data = pd.DataFrame({
        'ds': pd.to_datetime(future_dates),
        'Campaign_ID': [campaign_id] * len(future_dates)
    })
    
    for regressor, values in new_regressors.items():
        new_data[regressor] = values
        
    new_data['ds'] = pd.to_datetime(new_data['ds'])

    # Merge with the future DataFrame
    future = pd.merge(new_data, future, on=['ds', 'Campaign_ID'], how='inner')


    # Make predictions
    forecast = campaign_models[campaign_id].predict(future)

    # Collect predictions for all future dates
    predictions = forecast.loc[forecast['ds'].isin(future_dates), ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return predictions


def main():
    # Prompt the user for campaign, future dates, and regressor values
    campaign_id = input("Enter the Campaign ID: ")

    # Prompt for future dates
    future_dates_input = input("Enter the future dates you want to forecast (comma-separated, e.g. '2024-08-20,2024-08-21,2024-08-22'): ")
    future_dates = [pd.to_datetime(date.strip()) for date in future_dates_input.split(',')]

    new_regressors = {}
    for regressor in ['Audience_Reach', 'Platform_Type_Encoded']:
        regressor_values = [float(input(f"Enter the value for {regressor} on {date.strftime('%Y-%m-%d')}: ")) for date in future_dates]
        new_regressors[regressor] = regressor_values

    # (Assume campaign_models is defined elsewhere and contains your trained models)
    predictions = get_forecast(future_dates, new_regressors, campaign_id, campaign_models)

    # Print the predictions
    print("Predicted values for the target on the future dates:")
    print(predictions)


# Execute the main function
if __name__ == "__main__":
    main()



