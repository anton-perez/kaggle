import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/runner/kaggle/StudentsPerformance.csv')

print(df['math score'].mean())
print(df.columns)
completed = df['test preparation course'].apply(lambda entry: entry == 'completed')
no_comp = df['test preparation course'].apply(lambda entry: entry != 'completed')

print(df['math score'][completed].mean())
print(df['math score'][no_comp].mean())

#69.69553072625699
#64.0778816199377

df = pd.get_dummies(df,columns=['race/ethnicity','lunch','test preparation course', 'parental level of education'],drop_first=True)

training_df = df[:-3]
testing_df = df[-3:]
index = list(df.columns).index('math score')

training_array = np.array(training_df)
testing_array = np.array(testing_df)

y_train = training_array[:,index]
y_test = testing_array[:,index]

X_train = training_array[:,index+1:]
X_test = testing_array[:,index+1:]

#Linear Regressor
print('\nLinear Regression:')
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_test_predictions = regressor.predict(X_test)
print(y_test)
print(y_test_predictions)
