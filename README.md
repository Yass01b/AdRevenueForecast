# AdRevenueForecast is a time-series forecasting app designed to predict ad campaigns revenue using Facebook Prophet model.

---

## Exploratory Data Analysis (EDA)

### 1. Data Overview

The dataset represents advertising campaigns tracked daily over four months, retrieved from a MySQL database. The main columns include:

- **Campaign_ID**: Unique identifier for each campaign.
- **Date_Time**: Date and time of data recording, allowing for temporal analysis.
- **Platform_Type**: Advertising platform (e.g., Google, Facebook), enabling performance comparison across platforms.
- **Impressions**: Number of times the ad was displayed, a key measure of visibility.
- **Clicks**: Total clicks received, indicating user engagement.
- **Conversions**: Number of successful actions (e.g., purchases) resulting from the campaigns.
- **Cost**: Total expenditure on the campaign, essential for ROI calculations.
- **Revenue**: Total income generated, directly linked to campaign success.

These metrics provide a comprehensive view of campaign performance, allowing for in-depth analysis of advertising effectiveness.

---

### 2. Data Processing
Data was fetched from a MySQL database. We converted certain columns to numeric types where applicable. The columns `Campaign_ID`, `Platform_Type` were kept as categorical variables, and `Date_Time` was transformed into date format.

---

### 3. Revenue Analysis

#### Revenue by Platform Type Over Time (Monthly)
![Revenue by Platform Type (Monthly)](images/revenue_by_platform_monthly.png)
This plot shows the revenue distribution across different platform types over the months.

#### Revenue by Platform Type Over Time (Weekly)
![Revenue by Platform Type (Weekly)](images/revenue_by_platform_weekly.png)
This plot shows the revenue distribution across different platform types on a weekly basis.

#### Revenue by Campaign Over Time (Monthly)
![Revenue by Campaign (Monthly)](images/revenue_by_campaign_monthly.png)
This visual represents how different campaigns performed in terms of revenue month-wise.

#### Revenue by Campaign Over Time (Weekly)
![Revenue by Campaign (Weekly)](images/revenue_by_campaign_weekly.png)
Similar to above but at a weekly granularity.

---

### 4. Correlation Analysis
![Correlation Matrix](images/correlation_matrix.png)
This heatmap illustrates the correlations among various features in the dataset. Strong correlations can indicate where relationships exist.

---

### 5. Distribution of Features
Below are histograms showing the distribution of various numeric features in the dataset.

#### Histogram of Impressions
![Histogram of Impressions](images/histogram_of_Impressions.png)

##### Histogram of Clicks
![Histogram of Clicks](images/histogram_of_Clicks.png)

#### Histogram of Conversions
![Histogram of Conversions](images/histogram_of_Conversions.png)

#### Histogram of Tracked Ads
![Histogram of Tracked Ads](images/histogram_of_Tracked_Ads.png)

#### Histogram of Cost
![Histogram of Cost](images/histogram_of_Cost.png)

#### Histogram of Revenue
![Histogram of Revenue](images/histogram_of_Revenue.png)

#### Histogram of Average Frequency
![Histogram of Average Frequency](images/histogram_of_Average_Frequency.png)

#### Histogram of Audience Reach
![Histogram of Audience Reach](images/histogram_of_Audience_Reach.png)

#### Histogram of Unique Reach
![Histogram of Unique Reach](images/histogram_of_Unique_Reach.png)

#### Histogram of On Target Impressions
![Histogram of On Target Impressions](images/histogram_of_On_Target_Impressions.png)

#### Histogram of Audience Efficiency Rate
![Histogram of Audience Efficiency Rate](images/histogram_of_Audience_Efficiency_Rate.png)

#### Histogram of Percentage On Target
![Histogram of Percentage On Target](images/histogram_of_Percentage_On_Target.png)

---

### 6. Outlier Analysis
#### Box Plot of Impressions
![Box Plot of Impressions](images/box_plot_of_Impressions.png)
This box plot shows the distribution of impressions and highlights potential outliers.

#### Box Plot of Clicks
![Box Plot of Clicks](images/box_plot_of_Clicks.png)

#### Box Plot of Conversions
![Box Plot of Conversions](images/box_plot_of_Conversions.png)

#### Box Plot of Tracked Ads
![Box Plot of Tracked Ads](images/box_plot_of_Tracked_Ads.png)

#### Box Plot of Cost
![Box Plot of Cost](images/box_plot_of_Cost.png)

#### Box Plot of Revenue
![Box Plot of Revenue](images/box_plot_of_Revenue.png)

#### Box Plot of Average Frequency
![Box Plot of Average Frequency](images/box_plot_of_Average_Frequency.png)

#### Box Plot of Audience Reach
![Box Plot of Audience Reach](images/box_plot_of_Audience_Reach.png)

#### Box Plot of Unique Reach
![Box Plot of Unique Reach](images/box_plot_of_Unique_Reach.png)

#### Box Plot of On Target Impressions
![Box Plot of On Target Impressions](images/box_plot_of_On_Target_Impressions.png)

#### Box Plot of Audience Efficiency Rate
![Box Plot of Audience Efficiency Rate](images/box_plot_of_Audience_Efficiency_Rate.png)

#### Box Plot of Percentage On Target
![Box Plot of Percentage On Target](images/box_plot_of_Percentage_On_Target.png)

---

### 7. Additional Visualizations
#### Scatterplots of Key Features
![Scatter Plots](images/scatterplots.png)
These scatter plots visualize the relationships between different features.

---

### 8. Conclusion
This analysis provides valuable insights into revenue trends, correlations, distributions, and potential outliers, enabling better vision of data that we gonna use to build our Time-series model.

---