import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/eurisko-us/eurisko-us.github.io/master/files/debugging-help/processed_titanic_data.csv')

df = df[['Survived', 'Sex', 'Pclass', 'Fare', 'Age', 'SibSp']][:100]

def leave_one_out_cross_validation_accuracy(df, dependent_variable, k): 
  correct_classfications = 0
  total_classifications = len(df.to_numpy().tolist())
  for i in range(total_classifications):
    independent_df = df[[col for col in df.columns if col != dependent_variable]]
    dependent_df = df[dependent_variable]

    left_out = independent_df.iloc[[i]].to_numpy().tolist()[0]
    actual_classification = dependent_df.iloc[[i]].to_numpy().tolist()[0]

    independent = independent_df.drop([i]).reset_index(drop=True).to_numpy().tolist()
    dependent = dependent_df.drop([i]).reset_index(drop=True).to_numpy().tolist()
    
    knn = knearestclass(n_neighbors=k)
    knn = knn.fit(independent, dependent)
    predicted_classification = knn.predict([left_out])

    if predicted_classification == actual_classification:
      correct_classfications += 1

  return correct_classfications/total_classifications

k_vals = [1,3,5,10,15,20,30,40,50,75]
accuracies = [round(leave_one_out_cross_validation_accuracy(df, 'Survived', k), 2) for k in k_vals]

print(accuracies)

plt.style.use('bmh')
plt.plot(k_vals, accuracies)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.xticks(k_vals)
plt.title('Leave One Out Cross Validation')
plt.savefig('leave_one_out_accuracy.png')