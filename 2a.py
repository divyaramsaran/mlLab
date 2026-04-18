import pandas as pd
import numpy as np

# sample data
df = pd.DataFrame({
    'Age': [25, 30, np.nan, 22, 100],
    'Salary': [50000, np.nan, 60000, 55000, 999999]
})

print("Original Data:\n", df)

# a) Attribute Selection
df = df[['Age', 'Salary']]
print("\nAfter Attribute Selection:\n", df)

# b) Handling Missing Values
df.fillna(df.mean(), inplace=True)
print("\nAfter Handling Missing Values:\n", df)

# c) Discretization
df['Age'] = pd.cut(df['Age'], bins=2, labels=['Low', 'High'])
print("\nAfter Discretization:\n", df)

# d) Removing Outliers
df = df[df['Salary'] < 200000]
print("\nAfter Removing Outliers:\n", df)