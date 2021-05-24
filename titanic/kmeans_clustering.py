import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/eurisko-us/eurisko-us.github.io/master/files/debugging-help/processed_titanic_data.csv')
keep_cols = ["Sex", "Pclass", "Fare", "Age", "SibSp"]
new_df = df[keep_cols]
result_df = df[keep_cols + ['Survived']]

for col in new_df:
  new_df[col] = (new_df[col]-new_df[col].min()) / (new_df[col].max()-new_df[col].min())

k_vals = [k for k in range(1,26)]
error = []

data = new_df.to_numpy()

for k in k_vals:
  kmeans = KMeans(n_clusters=k).fit(data)
  error.append(kmeans.inertia_)

plt.style.use('bmh')
plt.plot(k_vals, error)
plt.xticks(k_vals)
plt.ylabel('sum squared distance from cluster center')
plt.xlabel('k')
plt.ylim(0, 425)
plt.savefig('kmeans_clustering_error.png')

kmeans = KMeans(n_clusters=4, random_state=0).fit(data)

def get_cluster_count(cluster_num):
  return list(kmeans.labels_).count(cluster_num)

result_df['cluster'] = list(kmeans.labels_)
result_df['count'] = result_df['cluster'].apply(get_cluster_count)
cluster_df = result_df.groupby(['cluster']).mean()
print(cluster_df)