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

df.loc[age_nan, ['Age']] = df['Age'][age_not_nan].mean()


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


features = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
columns = ['Survived'] + features
df = df[columns]



for var1 in features:
  for var2 in features[features.index(var1)+1:]:
    if not('Embarked=' in var1 and 'Embarked=' in var2):
      if not('CabinType=' in var1 and 'CabinType=' in var2):
        if not('SibSp' in var1 and 'SibSp' in var2):
          columns.append(var1 + " * " + var2)


interaction_features = columns[1:]

for var in interaction_features:
  if ' * ' in var:
    vars = var.split(' * ')
    df[var] = df[vars[0]]*df[vars[1]]


def convert_regressor_output_to_survival_value(n):
  if n < 0.5:
    return 0
  return 1

def get_accuracy(predictions, actual):
  correct_predictions = 0
  for n in range(len(predictions)):
    if predictions[n] == actual[n]:
      correct_predictions += 1
  return correct_predictions / len(predictions)


print("feature list = ", interaction_features, '\n')

selected_features = []
max_overall_testing_accuracy = 0
max_overall_training_accuracy = 0
while True:
  max_testing_accuracy = 0
  max_training_accuracy = 0
  new_feature = interaction_features[0]
  for current_feature in interaction_features:
    if current_feature not in selected_features:
      training_df = df[:500]
      testing_df = df[500:]

      training_df = training_df[['Survived'] + selected_features+[current_feature]]
      testing_df = testing_df[['Survived'] + selected_features +[current_feature]]

      training_array = np.array(training_df)
      testing_array = np.array(testing_df)

      y_train = training_array[:,0]
      y_test = testing_array[:,0]

      X_train = training_array[:,1:]
      X_test = testing_array[:,1:]

      regressor = LogisticRegression(max_iter=1000)
      regressor.fit(X_train, y_train)

      y_test_predictions = regressor.predict(X_test)
      y_train_predictions = regressor.predict(X_train)

      y_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_test_predictions]
      y_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_train_predictions]

      training_accuracy = get_accuracy(y_train_predictions, y_train)
      testing_accuracy =  get_accuracy(y_test_predictions, y_test)

      if testing_accuracy > max_testing_accuracy:
        max_testing_accuracy = testing_accuracy
        max_training_accuracy = training_accuracy
        new_feature = current_feature
  
  if max_testing_accuracy <= max_overall_testing_accuracy:
    break
  
  max_overall_testing_accuracy = max_testing_accuracy
  max_overall_training_accuracy = max_training_accuracy
  selected_features.append(new_feature)

  print("\n", selected_features, "\n")
  print("training", max_overall_training_accuracy)
  print("testing", max_overall_testing_accuracy, "\n")

print("FINISHED")


