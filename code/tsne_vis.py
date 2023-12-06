import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/Hum_ptmint_prot_el.csv'
data = pd.read_csv(file_path)

# Specify the columns for t-SNE
columns = ['prot_2_882', 'prot_2_847', 'prot_2_957', 'prot_2_1005', 'prot_2_172', 'prot_2_434',
    'prot_2_592', 'prot_1_173', 'prot_2_59', 'prot_1_340', 'prot_1_352', 'prot_2_1013',
    'prot_2_583', 'prot_1_56', 'prot_2_483', 'prot_2_861', 'wind_882', 'prot_1_829',
    'wind_792', 'prot_1_561', 'prot_1_291', 'prot_1_815', 'prot_2_951', 'prot_2_114',
    'prot_2_901', 'prot_2_107', 'wind_596', 'wind_532', 'prot_2_44', 'prot_1_374',
    'prot_2_372', 'prot_2_471', 'prot_2_319', 'wind_564', 'prot_2_996', 'wind_961',
    'wind_613', 'prot_2_573', 'wind_978', 'prot_2_193', 'prot_1_651', 'prot_1_547',
    'prot_1_501', 'prot_2_803', 'wind_495', 'wind_55', 'wind_659', 'prot_1_466', 'prot_2_6',
    'wind_420', 'wind_836', 'wind_865', 'wind_703', 'wind_850', 'wind_922', 'prot_1_453',
    'wind_878', 'prot_2_876', 'prot_1_887', 'wind_580', 'wind_793', 'prot_1_253',
    'prot_2_249', 'prot_2_739', 'prot_1_338', 'wind_393', 'wind_550', 'wind_778',
    'wind_220', 'prot_1_895', 'prot_2_810', 'prot_1_559', 'Is_int', 'PTM', 'Effect']  # Complete list of columns


"""
columns = ['prot_1_173', 'prot_1_340', 'prot_1_352',  'prot_1_56', 'wind_882', 'prot_1_829',
    'wind_792', 'prot_1_561', 'prot_1_291', 'prot_1_815', 'wind_596', 'wind_532', 'prot_2_44', 'prot_1_374',
    'wind_564', 'wind_961',
    'wind_613', 'wind_978', 'prot_1_651', 'prot_1_547',
    'prot_1_501',  'wind_495', 'wind_55', 'wind_659', 'prot_1_466',
    'wind_420', 'wind_836', 'wind_865', 'wind_703', 'wind_850', 'wind_922', 'prot_1_453',
    'wind_878','prot_1_887', 'wind_580', 'wind_793', 'prot_1_253',
    'prot_1_338', 'wind_393', 'wind_550', 'wind_778',
    'wind_220', 'prot_1_895', 'prot_1_559', 'Is_int', 'PTM', 'Effect']  # Complete list of columns

"""


# Filter the dataset
data_subset = data[columns]

data_subset.to_csv('H_ptm_short.csv')

# Replace 'Effect' values with categorical labels
data_subset.replace({'Effect': {0: 'Enhance', 1: 'Inhibit', 2: 'Induce'}}, inplace=True)
data_subset.replace({'PTM':{0:'Ac', 1:'Glyco', 2:'Me', 3:'Phos', 4:'Sumo', 5:'Ub'}}, inplace = True)
# Separate features and target
X = data_subset.drop('Effect', axis=1)
#X = X[X['PTM']!='Phos']
y = data_subset['Effect']
X = X.drop('PTM', axis=1)
X = X.drop('Is_int', axis=1)


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute t-SNE for 3 components
tsne = PCA(n_components=3)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Loop through categories to plot and create a legend
unique_labels = y.unique()
for label in unique_labels:
    indices = y == label
    ax.scatter(X_tsne[indices, 1], X_tsne[indices, 2], X_tsne[indices, 0], label=label, cmap = 'Accent', s = 1.5)

ax.set_title('3D PCA visualization')
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 1')

# ax.set_ylim(-10,40)
# ax.set_zlim(-40,20)
# ax.set_xlim(-40,30)
plt.legend()
plt.show()
plt.savefig('tsne_rep_3d.png')
