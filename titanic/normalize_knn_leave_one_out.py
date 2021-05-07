import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt

start_time = time.time()

def leave_one_out_cross_validation_accuracy(independent_df, dependent_df, k): 
  knn = knearestclass(n_neighbors=k)
  correct_classfications = 0
  total_classifications = len(independent_df)
  print(k)
  for i in range(total_classifications):
    indep_copy = list(independent_df)
    dep_copy = list(dependent_df)

    left_out = indep_copy[i]
    actual_classification = dep_copy[i]

    indep_copy.pop(i)
    dep_copy.pop(i)
    
    knn = knn.fit(indep_copy, dep_copy)
    predicted_classification = knn.predict([left_out])

    if predicted_classification == actual_classification:
      correct_classfications += 1

  return correct_classfications/total_classifications


df = pd.read_csv('https://raw.githubusercontent.com/eurisko-us/eurisko-us.github.io/master/files/debugging-help/processed_titanic_data.csv')

df = df[['Survived', 'Sex', 'Pclass', 'Fare', 'Age', 'SibSp']][:100]

simple_df = df.copy()
minmax_df = df.copy()
z_score_df = df.copy()

features = ["Sex", "Pclass", "Fare", "Age","SibSp"]
for feature in features:
  simple_df[feature] = simple_df[feature]/simple_df[feature].max()
  minmax_df[feature] = (minmax_df[feature]-minmax_df[feature].min()) / (minmax_df[feature].max()-minmax_df[feature].min())
  z_score_df[feature] = (z_score_df[feature]-z_score_df[feature].mean()) / z_score_df[feature].std()

dependent_df = df['Survived'].to_numpy().tolist()

independent_df = df[[col for col in df.columns if col != 'Survived']].to_numpy().tolist()

indep_simple_df = simple_df[[col for col in df.columns if col != 'Survived']].to_numpy().tolist()

indep_minmax_df = minmax_df[[col for col in df.columns if col != 'Survived']].to_numpy().tolist()

indep_z_score_df = z_score_df[[col for col in df.columns if col != 'Survived']].to_numpy().tolist()


k_vals = [2*n+1 for n in range(50)]
accuracies = [{
  'unnormalized':leave_one_out_cross_validation_accuracy(independent_df, dependent_df, k),
  'simple scaling':leave_one_out_cross_validation_accuracy(indep_simple_df, dependent_df, k),
  'min-max': leave_one_out_cross_validation_accuracy(indep_minmax_df, dependent_df, k), 
  'z-scoring': leave_one_out_cross_validation_accuracy(indep_z_score_df, dependent_df, k)} 
  for k in k_vals]


unnormalized_accuracies = [i["unnormalized"] for i in accuracies]
simple_accuracies = [i["simple scaling"] for i in accuracies]
minmax_accuracies = [i["min-max"] for i in accuracies]
z_score_accuracies = [i["z-scoring"] for i in accuracies]

plt.style.use('bmh')
plt.plot(k_vals, unnormalized_accuracies, label="unnormalized")
plt.plot(k_vals, simple_accuracies, label="simple scaling")
plt.plot(k_vals, minmax_accuracies, label="min-max")
plt.plot(k_vals, z_score_accuracies, label="z-scoring")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Leave One Out Cross Accuracy for Various Normalization')
plt.legend()
plt.savefig('normalized_leave_one_out_accuracy.png')

end_time = time.time()
print('time taken:', end_time - start_time)