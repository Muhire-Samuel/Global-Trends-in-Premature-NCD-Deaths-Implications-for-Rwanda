import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# Preview the data
df.head()
# Check missing values
df.isnull().sum()

# Drop irrelevant or fully missing columns (e.g., comments, IDs)
df.drop(columns=["FactComments", "FactValueTranslationID"], inplace=True)

# Convert date fields
df['DateModified'] = pd.to_datetime(df['DateModified'])

# Convert numeric fields
df['FactValueNumeric'] = pd.to_numeric(df['FactValueNumeric'], errors='coerce')

# Fill or drop missing numerical data
df['FactValueNumeric'].fillna(df['FactValueNumeric'].median(), inplace=True)
import numpy as np

# Example: IQR method to remove outliers
Q1 = df['FactValueNumeric'].quantile(0.25)
Q3 = df['FactValueNumeric'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['FactValueNumeric'] >= Q1 - 1.5 * IQR) & (df['FactValueNumeric'] <= Q3 + 1.5 * IQR)]
# Encode 'Sex' and 'Location' if needed
df_encoded = pd.get_dummies(df[['Dim1', 'Location']], drop_first=True)
df = pd.concat([df, df_encoded], axis=1)

df.describe()
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of mortality rates
sns.histplot(df['FactValueNumeric'], kde=True)
plt.title('Distribution of NCD Mortality Rates')
plt.show()

# Compare regions
plt.figure(figsize=(10, 6))
sns.boxplot(x='ParentLocation', y='FactValueNumeric', data=df)
plt.title('NCD Mortality Rates by WHO Region')
plt.xticks(rotation=45)
plt.show()
# Time series trend for Rwanda
rwanda = df[df['Location'] == 'Rwanda']
plt.plot(rwanda['Period'], rwanda['FactValueNumeric'])
plt.title('Trend of NCD Mortality in Rwanda Over Years')
plt.xlabel('Year')
plt.ylabel('Mortality Rate')
plt.grid(True)
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select features
features = df[['FactValueNumeric']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')

df['Cluster'] = kmeans.fit_predict(features_scaled)

from sklearn.metrics import silhouette_score

score = silhouette_score(features_scaled, df['Cluster'])
print(f"Silhouette Score: {score:.2f}")
def clean_data(df):
    df = df.copy()
    df['FactValueNumeric'] = pd.to_numeric(df['FactValueNumeric'], errors='coerce')
    df['FactValueNumeric'].fillna(df['FactValueNumeric'].median(), inplace=True)
    return df
def high_mortality_countries(df, threshold):
    """
    Returns a DataFrame with countries that exceed the given mortality threshold.
    """
    high_df = df[df['FactValueNumeric'] > threshold]
    print(f"Found {high_df['Location'].nunique()} countries above {threshold} deaths per 100,000.")
    return high_df[['Location', 'FactValueNumeric', 'Period']].sort_values(by='FactValueNumeric', ascending=False)

# Example usage:
high_mortality = high_mortality_countries(df, threshold=900)
high_mortality.head()
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Select features and scale them
features = df[['FactValueNumeric']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(features_scaled)

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['FactValueNumeric'], df['Period'], c=df['DBSCAN_Cluster'], cmap='viridis')
plt.title('DBSCAN Clustering of Countries by Mortality Rate')
plt.xlabel('NCD Mortality Rate')
plt.ylabel('Year')
plt.colorbar(label='Cluster Label')
plt.show()

# Inspect cluster assignments
df['DBSCAN_Cluster'].value_counts()
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Prepare Rwanda data
rwanda_ts = df[(df['Location'] == 'Rwanda') & (df['Period type'] == 'Year')]
rwanda_ts = rwanda_ts[['Period', 'FactValueNumeric']].dropna().sort_values('Period')

# Ensure datetime index
rwanda_ts['Period'] = pd.to_datetime(rwanda_ts['Period'], format='%Y')
rwanda_ts.set_index('Period', inplace=True)

# Fit ARIMA model (p=1, d=1, q=1 as a basic choice)
model = ARIMA(rwanda_ts['FactValueNumeric'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 5 years
forecast = model_fit.forecast(steps=5)

# Plot actual and forecast
plt.figure(figsize=(10, 6))
plt.plot(rwanda_ts, label='Observed')
plt.plot(pd.date_range(start=rwanda_ts.index[-1], periods=6, freq='Y')[1:], forecast, label='Forecast', linestyle='--')
plt.title('Forecast of NCD Mortality in Rwanda')
plt.xlabel('Year')
plt.ylabel('Mortality Rate')
plt.legend()
plt.grid(True)
plt.show()




# Load the dataset


# Preview
print("Initial shape:", df.shape)
print(df.head(3))

# Drop irrelevant or unused columns
cols_to_drop = [
    'FactValueTranslationID', 'FactComments', 'Language', 'Dim2 type', 'Dim2', 'Dim2ValueCode',
    'Dim3 type', 'Dim3', 'Dim3ValueCode', 'FactValueNumericPrefix',
    'FactValueNumericLowPrefix', 'FactValueNumericHighPrefix', 'Value'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Convert 'DateModified' to datetime
df['DateModified'] = pd.to_datetime(df['DateModified'], errors='coerce')

# Convert numeric values to float
df['FactValueNumeric'] = pd.to_numeric(df['FactValueNumeric'], errors='coerce')
df['FactValueNumericLow'] = pd.to_numeric(df['FactValueNumericLow'], errors='coerce')
df['FactValueNumericHigh'] = pd.to_numeric(df['FactValueNumericHigh'], errors='coerce')

# Handle missing numeric data
df['FactValueNumeric'].fillna(df['FactValueNumeric'].median(), inplace=True)

# Drop rows with no location or year info
df.dropna(subset=['Location', 'Period'], inplace=True)

# Convert Period to integer (in case it's not)
df['Period'] = pd.to_numeric(df['Period'], errors='coerce').astype('Int64')

# Keep only relevant locations (optional: drop regions like "World" or "Africa")
df = df[df['Location type'] == 'Country']

# Standardize column names (optional)
df.rename(columns={
    'FactValueNumeric': 'NCD_Mortality_Rate',
    'Dim1': 'Sex',
    'Period': 'Year'
}, inplace=True)

# Optional: filter out extreme outliers using IQR
Q1 = df['NCD_Mortality_Rate'].quantile(0.25)
Q3 = df['NCD_Mortality_Rate'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['NCD_Mortality_Rate'] >= Q1 - 1.5 * IQR) & (df['NCD_Mortality_Rate'] <= Q3 + 1.5 * IQR)]

# Final shape and info
print("Cleaned shape:", df.shape)
print(df.dtypes)

# Optional: save cleaned data to CSV
df.to_csv("ncd_deaths_cleaned.csv", index=False)
