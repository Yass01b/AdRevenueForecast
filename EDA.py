import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, chi2
import seaborn as sns
import joblib
from prophet import Prophet
import mysql.connector
from scipy.stats import skew, kurtosis  
import warnings
warnings.filterwarnings("ignore") 
from dotenv import load_dotenv
import os
import mysql.connector


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
# Revenue by Platform Type Over Time (Monthly)
platform_revenue = df.groupby(['Platform_Type', df['Date_Time'].dt.to_period('M')])['Revenue'].sum().reset_index()

# Convert the 'Date_Time' column to a numeric representation (e.g., timestamp)
platform_revenue['Date_Time'] = platform_revenue['Date_Time'].dt.to_timestamp()

# Plot the revenue for each platform type over time
plt.figure(figsize=(12, 6))
for platform in platform_revenue['Platform_Type'].unique():
    platform_data = platform_revenue[platform_revenue['Platform_Type'] == platform]
    plt.plot(platform_data['Date_Time'], platform_data['Revenue'], label=platform)
plt.xticks(rotation=60)
plt.xlabel('Date (Monthly)')
plt.ylabel('Revenue')
plt.title('Revenue by Platform Type Over Time (Monthly)')
plt.legend()
plt.show()

# Revenue by Platform Type Over Time (Weekly)
platform_revenue = df.groupby(['Platform_Type', df['Date_Time'].dt.to_period('W')])['Revenue'].sum().reset_index()

# Convert the 'Date_Time' column to a numeric representation (e.g., timestamp)
platform_revenue['Date_Time'] = platform_revenue['Date_Time'].dt.to_timestamp()

# Plot the revenue for each platform type over time
plt.figure(figsize=(12, 6))
for platform in platform_revenue['Platform_Type'].unique():
    platform_data = platform_revenue[platform_revenue['Platform_Type'] == platform]
    plt.plot(platform_data['Date_Time'], platform_data['Revenue'], label=platform)
plt.xticks(rotation=60)
plt.xlabel('Date (Weekly)')
plt.ylabel('Revenue')
plt.title('Revenue by Platform Type Over Time (Weekly)')
plt.legend()
plt.show()

# Revenue by Campaign Over Time (Monthly)
campaign_revenue = df.groupby(['Campaign_ID', df['Date_Time'].dt.strftime('%Y-%m')])['Revenue'].sum().reset_index()

# Rename the columns for better readability
campaign_revenue.columns = ['Campaign_ID', 'Month', 'Revenue']

# Plot the revenue for each campaign over time
plt.figure(figsize=(12, 6))
for campaign in campaign_revenue['Campaign_ID'].unique():
    campaign_data = campaign_revenue[campaign_revenue['Campaign_ID'] == campaign]
    plt.plot(campaign_data['Month'], campaign_data['Revenue'], label=f'Campaign {campaign}')
plt.xticks(rotation=60)
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Revenue by Campaign Over Time (Monthly)')
plt.legend()
plt.show()

# Revenue by Campaign Over Time (Weekly)
campaign_revenue = df.groupby(['Campaign_ID', df['Date_Time'].dt.isocalendar().week])['Revenue'].sum().reset_index()

# Rename the columns for better readability
campaign_revenue.columns = ['Campaign_ID', 'Week', 'Revenue']

# Plot the revenue for each campaign over time
plt.figure(figsize=(12, 6))
for campaign in campaign_revenue['Campaign_ID'].unique():
    campaign_data = campaign_revenue[campaign_revenue['Campaign_ID'] == campaign]
    plt.plot(campaign_data['Week'], campaign_data['Revenue'], label=f'Campaign {campaign}')
plt.xticks(rotation=60)
plt.xlabel('Week')
plt.ylabel('Revenue')
plt.title('Revenue by Campaign Over Time (Weekly)')
plt.legend()
plt.show()

# Let's create a copy of our data
copied_df = df.copy()

# Drop Campaign_ID feature
copied_df = copied_df.drop('Campaign_ID', axis=1)

#One-Hot encoding for Platform_Type feature

# Instantiate the Label Encoder Object
label_encoder = LabelEncoder()
# Fit and transform the column
encoded_data = label_encoder.fit_transform(copied_df['Platform_Type'])
# Create a new DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=['Platform_Type_Encoded'])

# Concatenate the encoded DataFrame with the copied DataFrame
copied_df = pd.concat([copied_df, encoded_df], axis=1)

# Let's explore the correlation Matrix heatmap
data = copied_df[['Date_Time', 'Impressions', 'Clicks', 'Conversions', 'Tracked_Ads', 'Cost', 'Revenue', 'Average_Frequency', 'Audience_Reach', 'Unique_Reach', 'On_Target_Impressions', 'Audience_Efficiency_Rate', 'Percentage_On_Target', 'Platform_Type_Encoded']]
corr = data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
ax.set_title('Correlation Matrix')
plt.show()

#
data = copied_df[['Date_Time', 'Impressions', 'Clicks', 'Conversions', 'Tracked_Ads', 'Cost', 'Revenue', 'Average_Frequency', 'Audience_Reach', 'Unique_Reach', 'On_Target_Impressions', 'Audience_Efficiency_Rate', 'Percentage_On_Target', 'Platform_Type_Encoded']]

data_aggregated = data.groupby('Date_Time')['Revenue'].sum()
data_aggregated.plot(kind='line', figsize=(15, 5), color='blue', title='Revenue over time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show()


# Visualize the data distribution
for column in data.columns:
    feature_data = data[column]
    
    # Check the data type of the feature
    if feature_data.dtype in ['float64', 'int64']:
        plt.figure(figsize=(8, 6))
        feature_data.hist(bins=30, density=True, alpha=0.5)
        
        # Plot the normal distribution curve
        mean = feature_data.mean()
        std_dev = feature_data.std()
        x = np.linspace(feature_data.min(), feature_data.max(), 100)
        plt.plot(x, norm.pdf(x, mean, std_dev), 'r-', lw=2)
        
        plt.title(f'Histogram of {column}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()

        # Calculate the descriptive statistics
        mean = feature_data.mean()
        median = feature_data.median()
        std_dev = feature_data.std()
        skewness = skew(feature_data)
        kurtosis_val = kurtosis(feature_data)

        print(f'Feature: {column}')
        print(f'Mean: {mean:.2f}')
        print(f'Median: {median:.2f}')
        print(f'Standard Deviation: {std_dev:.2f}')
        print(f'Skewness: {skewness:.2f}')
        print(f'Kurtosis: {kurtosis_val:.2f}')
        
        # Calculate the Z-score and percentage of data within 1, 2, and 3 standard deviations
        z_score_1 = (feature_data - mean) / std_dev
        pct_1_std = np.round(100 * np.sum(np.abs(z_score_1) <= 1) / len(feature_data), 2)
        pct_2_std = np.round(100 * np.sum(np.abs(z_score_1) <= 2) / len(feature_data), 2)
        pct_3_std = np.round(100 * np.sum(np.abs(z_score_1) <= 3) / len(feature_data), 2)
        
        print(f'Percentage of data within 1 standard deviation: {pct_1_std}%')
        print(f'Percentage of data within 2 standard deviations: {pct_2_std}%')
        print(f'Percentage of data within 3 standard deviations: {pct_3_std}%')
        print('---')



# let's check the Outliers

# Assuming 'data' is your DataFrame
for column in data.columns:
    feature_data = data[column]
    if feature_data.dtype in ['float64', 'int64']:
        # Z-score method
        z_scores = (feature_data - feature_data.mean()) / feature_data.std()
        outliers_z = np.where(np.abs(z_scores) > 3)

        # Tukey's method
        q1 = feature_data.quantile(0.25)
        q3 = feature_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_tukey = np.where((feature_data < lower_bound) | (feature_data > upper_bound))

        print(f"Feature: {column}")
        print(f"Outliers (Z-score): {len(outliers_z[0])} out of {len(feature_data)}")
        print(f"Outliers (Tukey's method): {len(outliers_tukey[0])} out of {len(feature_data)}")

        # Visualize the outliers
        plt.figure(figsize=(8, 6))
        feature_data.plot(kind='box')
        plt.title(f"Box Plot of {column}")
        plt.show()       
        

data = copied_df[['Impressions', 'Clicks', 'Conversions', 'Tracked_Ads', 'Cost', 'Revenue', 'Average_Frequency', 'Audience_Reach', 'Unique_Reach', 'On_Target_Impressions', 'Audience_Efficiency_Rate', 'Percentage_On_Target', 'Platform_Type_Encoded']]

# Create a figure with 5 rows and 3 columns of subplots
fig, axs = plt.subplots(5, 3, figsize=(16, 18))

# Iterate through the features and create a scatterplot for each
feature_names = ['Impressions', 'Clicks', 'Conversions', 'Tracked_Ads', 'Cost', 'Revenue', 'Average_Frequency', 'Audience_Reach', 'Unique_Reach', 'On_Target_Impressions', 'Audience_Efficiency_Rate', 'Percentage_On_Target', 'Platform_Type_Encoded']
row, col = 0, 0

for feature in feature_names:
    ax = axs[row, col]
    ax.scatter(data.index, data[feature], c='blue', marker='*', alpha=0.5)
    ax.set_title(feature)
    ax.set_xlabel('Index')
    ax.set_ylabel(feature)
    col += 1
    if col > 2:
        col = 0
        row += 1

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Display the figure
plt.show()
