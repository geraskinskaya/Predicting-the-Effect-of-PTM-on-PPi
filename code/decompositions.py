import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE



data = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/all_means_final.csv')
print(len(data))
data = data.drop_duplicates()
print(len(data))
uns = data.iloc[:, 1].values.tolist()

data = data.iloc[:, 2:]


#pca

X = data
std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(X)

pca = PCA(n_components = 200)
x_t = pca.fit_transform(scaled_df)
N = list(range(200))
X_new = pd.DataFrame(x_t, columns=[f'feat{n}' for n in N])
X_new['un'] = uns

X_new.to_csv('Pca_composed.csv')


umap_transformer_subset = umap.UMAP(n_neighbors=6, n_components=8, random_state=42)
umap_embedding_subset = umap_transformer_subset.fit_transform(X)
j = range(8)
df1 = pd.DataFrame(umap_embedding_subset, columns=[f'u_{a}' for a in j])

df1['un'] = uns

df1.to_csv('umaps.csv')

X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

print(TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit(X).get_feature_names_out())

df3 = pd.DataFrame(X_embedded, columns = [f'ts{a}' for a in range(3)])
df3['un'] = uns
df3.to_csv('tsne.csv')