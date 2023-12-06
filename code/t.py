import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Load the dataset
file_path = 'C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/F_ptmint_prot_wind_enc.csv'
data = pd.read_csv(file_path)

# Specify the columns for t-SNE
columns = ['prot_1_173', 'prot_1_340', 'prot_1_352', 'prot_1_56', 'wind_882', 'prot_1_829',
           'wind_792', 'prot_1_561', 'prot_1_291', 'prot_1_815', 'wind_596', 'wind_532', 'prot_2_44', 'prot_1_374',
           'wind_564', 'wind_961', 'wind_613', 'wind_978', 'prot_1_651', 'prot_1_547',
           'prot_1_501', 'wind_495', 'wind_55', 'wind_659', 'prot_1_466', 'wind_420', 'wind_836', 'wind_865',
           'wind_703', 'wind_850', 'wind_922', 'prot_1_453', 'wind_878', 'prot_1_887', 'wind_580', 'wind_793',
           'prot_1_253', 'prot_1_338', 'wind_393', 'wind_550', 'wind_778', 'wind_220', 'prot_1_895',
           'prot_1_559', 'Is_int', 'PTM', 'Effect']

# Filter the dataset
data_subset = data[columns]

# Replace 'Effect' values with categorical labels
data_subset.replace({'Effect': {0: 'Enhance', 1: 'Inhibit', 2: 'Induce'}}, inplace=True)

# Separate features and target
X = data_subset.drop(['Effect', 'PTM', 'Is_int'], axis=1)
y = data_subset['PTM']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute t-SNE for 2 components
tsne = PCA(n_components=2)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting in 2D
plt.figure(figsize=(12, 8))

# Loop through categories to plot and create a legend
unique_labels = y.unique()
for label in unique_labels:
    indices = y == label
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=label, cmap='Accent', s=1.5)

plt.title('2D t-SNE visualization')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.legend()
plt.show()
