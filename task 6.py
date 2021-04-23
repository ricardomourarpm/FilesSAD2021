import numpy as np
import pandas as pd

data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

Weight = data.Weight

data = data.drop(columns='Weight')

types = data.NObeyesdad.unique()

not_obese = types[[0,4]]

data['Obese'] = [da in not_obese for da in data.NObeyesdad]

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score,silhouette_samples

data_dummies = pd.get_dummies(data)
scaler = MinMaxScaler().fit(data_dummies)
preprocessed_data = scaler.fit_transform(data_dummies)

for i in range(10):
    clustering = KMeans(n_clusters=18,random_state=i).fit(preprocessed_data)
    labels = clustering.labels_
    print(i,silhouette_score(preprocessed_data,labels=labels))

trial_number = input()

clustering = KMeans(n_clusters=18,random_state=8).fit(preprocessed_data)
labels = clustering.labels_
print(silhouette_score(preprocessed_data,labels=labels))

sample_silhouette = silhouette_samples(preprocessed_data,labels=labels)

data['clusters'] = labels
data['silhouette'] = sample_silhouette

group = data.groupby('clusters')

print(np.argmax(group.mean().silhouette))

analysis_group = group.get_group(1)

analysis_group.describe()

import matplotlib.pyplot as plt
analysis_group.MTRANS.hist()
plt.show()

analysis_group.CALC.hist()
plt.show()

analysis_group.NObeyesdad.hist()
plt.show()

analysis_group.Age.hist()
plt.show()

analysis_group.hist()
plt.show()

from sklearn.manifold import MDS

embedding = MDS(n_components=2,verbose=1,max_iter=100,n_init=2)

data_emb = embedding.fit_transform(data_dummies[0:])

for cluster in data['clusters'].unique():
    _ = plt.scatter(data_emb[0:][data.clusters[0:]==cluster][:,0], data_emb[0:][data.clusters[0:]==cluster][:,1], cmap=plt.cm.Spectral,
            label='Cluster'+str(cluster)
            )
plt.legend()
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=38)
data_pca = pca.fit_transform(data_dummies)

embedding = MDS(n_components=2,verbose=1,max_iter=100,n_init=2)

data_emb = embedding.fit_transform(data_pca)

for cluster in data['clusters'].unique():
    _ = plt.scatter(data_emb[0:][data.clusters[0:]==cluster][:,0], data_emb[0:][data.clusters[0:]==cluster][:,1], cmap=plt.cm.Spectral,
            label='Cluster'+str(cluster)
            )
plt.legend()
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(data_dummies[0:100],'ward') # só para os primeiros 100

labelList = range(1, len(data_dummies[0:100])+1)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            truncate_mode='lastp',
            p=105,
            labels=labelList,
            color_threshold=8,
            distance_sort='ascending',
            show_leaf_counts=True)
plt.show()

# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/