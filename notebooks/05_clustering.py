# Clustering example

from sklearn.cluster import KMeans
import seaborn as sns

data = sns.load_dataset("iris")

X = data.drop("species", axis=1)

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

print(kmeans.labels_)
