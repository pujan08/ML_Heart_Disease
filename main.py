import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df=pd.read_csv("https://raw.githubusercontent.com/pujan08/ML_Heart_Disease/main/heart_disease.csv")
print(df)
print(df.info())
for i in range(len(df.columns)):
    column_name = df.columns[i]
    missing_data = df[column_name].isna().sum()
    perc = missing_data / len(df) * 100
    print(f'Feature {i+1} >> Missing entries: {missing_data}  |  Percentage: {round(perc, 2)}')

plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

imputer.fit(df)
#Fit the imputer model on dataset to calculate statistic for each column
column_statistics = imputer.statistics_
print(column_statistics)

#Trained imputer model is applied to dataset to create a copy of dataset
# with all filled missing values from the calculated statistic using transform( )
df_filled = pd.DataFrame(imputer.transform(df), columns=df.columns)
missing_values_after_imputation = df_imputed.isna().sum().sum()
print(df_imputed)

if missing_values_after_imputation == 0:
    print("Sanity Check Passed: No missing values after imputation.")
else:
    print(f"Sanity Check Failed: There are still {missing_values_after_imputation} missing values after imputation.")

print(missing_values_after_imputation)
plt.figure(figsize=(10,6))
sns.heatmap(df_imputed.isna(), cbar=False, cmap='viridis', yticklabels=False)

