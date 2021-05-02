import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/eurisko-us/eurisko-us.github.io/master/files/datasets/data-scientist-hiring.csv')

print(df['training_hours'].mean())
print(df['target'].mean())

print(df.groupby(['city']).count())
print(df.groupby(['city']).count().sort_values(by='target'))

print(df.groupby(['city']).count()[50:100])

# less_than_ten = df['company_size'].apply(lambda entry: entry < 10)
# less_than_hundred = df['company_size'].apply(lambda entry: entry < 100)

# print(df['target'][less_than_ten].count())
# print(df['target'][less_than_hundred].count())

print(df.groupby(['company_size']).count())