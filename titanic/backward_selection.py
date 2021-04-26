import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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

def get_df_accuracy(df, columns):
  training_df = df[:500]
  testing_df = df[500:]

  training_df = training_df[columns]
  testing_df = testing_df[columns]

  training_array = np.array(training_df)
  testing_array = np.array(testing_df)

  y_train = training_array[:,0]
  y_test = testing_array[:,0]

  X_train = training_array[:,1:]
  X_test = testing_array[:,1:]

  regressor = LogisticRegression(max_iter=100)
  regressor.fit(X_train, y_train)

  y_test_predictions = regressor.predict(X_test)
  y_train_predictions = regressor.predict(X_train)

  y_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_test_predictions]
  y_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_train_predictions]

  training_accuracy = get_accuracy(y_train_predictions, y_train)
  testing_accuracy =  get_accuracy(y_test_predictions, y_test)

  return {'test':testing_accuracy, 'train':training_accuracy}


print("feature list = ", interaction_features, '\n')

selected_features = interaction_features.copy()
removed_indices = []
base_testing_accuracy = get_df_accuracy(df, ['Survived']+interaction_features)['test']

for index in range(len(interaction_features)):
  current_feature = interaction_features[index]

  print('candidate for removal: ', current_feature, ' (index '+str(index)+')')
  # training_df = df[:500]
  # testing_df = df[500:]

  columns = ['Survived'] + selected_features
  # base_training_df = training_df[columns]
  # base_testing_df = testing_df[columns]
  
  columns.remove(current_feature)
  removed_training_accuracy = get_df_accuracy(df, columns)['train']
  removed_testing_accuracy = get_df_accuracy(df, columns)['test']
  # removed_training_df = training_df[columns]
  # removed_testing_df = testing_df[columns]

  # base_training_array = np.array(base_training_df)
  # removed_training_array = np.array(removed_training_df)
  # base_testing_array = np.array(base_testing_df)
  # removed_testing_array = np.array(removed_testing_df)

  # y_base_train = base_training_array[:,0]
  # y_removed_train = removed_training_array[:,0]
  # y_base_test = base_testing_array[:,0]
  # y_removed_test = removed_testing_array[:,0]

  # X_base_train = base_training_array[:,1:]
  # X_removed_train = removed_training_array[:,1:]
  # X_base_test = base_testing_array[:,1:]
  # X_removed_test = removed_testing_array[:,1:]

  # base_regressor = LogisticRegression(max_iter=100)
  # base_regressor.fit(X_base_train, y_base_train)

  # removed_regressor = LogisticRegression(max_iter=100)
  # removed_regressor.fit(X_removed_train, y_removed_train)

  # y_base_test_predictions = base_regressor.predict(X_base_test)
  # y_base_train_predictions = base_regressor.predict(X_base_train)
  # y_removed_test_predictions = removed_regressor.predict(X_removed_test)
  # y_removed_train_predictions = removed_regressor.predict(X_removed_train)

  # y_base_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_base_test_predictions]
  # y_base_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_base_train_predictions]

  # y_removed_test_predictions = [convert_regressor_output_to_survival_value(n) for n in y_removed_test_predictions]
  # y_removed_train_predictions = [convert_regressor_output_to_survival_value(n) for n in y_base_train_predictions]

  # base_training_accuracy = get_accuracy(y_base_train_predictions, y_base_train)
  # base_testing_accuracy =  get_accuracy(y_base_test_predictions, y_base_test)
  # removed_training_accuracy = get_accuracy(y_removed_train_predictions, y_removed_train)
  # removed_testing_accuracy =  get_accuracy(y_removed_test_predictions, y_removed_test)

  print("training:", removed_training_accuracy)
  print("testing:", removed_testing_accuracy)

  if base_testing_accuracy <= removed_testing_accuracy:
    removed_indices.append(index)
    selected_features.remove(current_feature)
    base_testing_accuracy = removed_testing_accuracy
    print("removed")
  else:
    print("kept")
  
  print("baseline testing accuracy: ", base_testing_accuracy)
  print("removed indices: ", removed_indices, "\n") 

print("FINISHED")

