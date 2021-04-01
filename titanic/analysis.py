import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/runner/kaggle/titanic/dataset_of_knowns.csv')

desired_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[desired_cols]

#Sex
def convert_sex_to_int(sex):
  if sex == 'male':
    return 0
  elif sex == 'female':
    return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)


#Age
age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))

mean_age = df['Age'][age_not_nan].mean()
df['Age'][age_nan] = mean_age


#SipSp
def indicator_greater_than_zero(x):
  if x > 0:
    return 1
  else:
    return 0

df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero)


#Parch
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)
del df['Parch']


#CabinType
df['Cabin']= df['Cabin'].fillna('None')

def cabin_type(cabin):
  if cabin != 'None':
    return cabin[0]
  return cabin


df['CabinType'] = df['Cabin'].apply(cabin_type)

for cabin_type in df['CabinType'].unique():
  dummy_variable_name = 'CabinType={}'.format(cabin_type)
  dummy_variable_values = df['CabinType'].apply(lambda entry: int(entry==cabin_type))
  df[dummy_variable_name] = dummy_variable_values

del df['CabinType']

#Embarked
df['Embarked'] = df['Embarked'].fillna('None')

for embark in df['Embarked'].unique():
  dummy_variable_name = 'Embarked={}'.format(embark)
  dummy_variable_values = df['Embarked'].apply(lambda entry: int(entry==embark))
  df[dummy_variable_name] = dummy_variable_values

del df['Embarked']


features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
columns = ['Survived'] + features_to_use
df = df[columns]


training_df = df[:500]
testing_df = df[500:]

training_array = np.array(training_df)
testing_array = np.array(testing_df)

y_train = training_array[:,0]
y_test = testing_array[:,0]

X_train = training_array[:,1:]
X_test = testing_array[:,1:]

#Linear Regressor
print('\nLinear Regression:')
regressor = LinearRegression()
regressor.fit(X_train, y_train)

coefficients = {}
feature_columns = training_df.columns[1:]
feature_coefficients = regressor.coef_

for n in range(len(feature_columns)):
  column = feature_columns[n]
  coefficient = feature_coefficients[n]
  coefficients[column] = coefficient

y_test_predictions = regressor.predict(X_test)
y_train_predictions = regressor.predict(X_train)

def convert_regressor_output_to_survival_value(n):
  if n < 0.5:
    return 0
  return 1

y_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_test_predictions]
y_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_train_predictions]

def get_accuracy(predictions, actual):
  correct_predictions = 0
  for n in range(len(predictions)):
    if predictions[n] == actual[n]:
      correct_predictions += 1
  return correct_predictions / len(predictions)


print("\n", "features:", features_to_use, "\n")
print("training accuracy:", round(get_accuracy(y_train_predictions, y_train), 4))
print("testing accuracy:", round(get_accuracy(y_test_predictions, y_test), 4), "\n")

coefficients['constant'] = regressor.intercept_
print({k: round(v, 4) for k, v in coefficients.items()})

#Logistic Regressor
print('\nLogistic Regression:')
regressor = LogisticRegression(max_iter=1000)
regressor.fit(X_train, y_train)

coefficients = {}
feature_columns = list(training_df.columns[1:])
feature_coefficients = list(regressor.coef_)[0]


for n in range(len(feature_columns)):
  column = feature_columns[n]
  coefficient = feature_coefficients[n]
  coefficients[column] = coefficient

y_test_predictions = regressor.predict(X_test)
y_train_predictions = regressor.predict(X_train)

def convert_regressor_output_to_survival_value(n):
  if n < 0.5:
    return 0
  return 1

y_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_test_predictions]
y_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_train_predictions]


print("\n", "features:", features_to_use, "\n")
print("training accuracy:", round(get_accuracy(y_train_predictions, y_train), 4))
print("testing accuracy:", round(get_accuracy(y_test_predictions, y_test), 4), "\n")

coefficients['constant'] = list(regressor.intercept_)[0]
print({k: round(v, 4) for k, v in coefficients.items()})