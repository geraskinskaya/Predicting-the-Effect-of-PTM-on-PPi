import numpy as np



import os
import pandas as pd
import numpy as np
from pyfaidx import Fasta

def extract_id(header):
    return header.split('|')[1]


dirpath = os.path.dirname('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/')
filepath = os.path.join(dirpath, 'Correct_filtered.csv')
data = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/PTM experimental evidence.csv')
test_data = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/Bett.csv')
#data = data[data['Organism'] == 'Human']
l1 = data['Uniprot'].values.tolist()
print(len(l1))
l2 = data['Int_uniprot'].values.tolist()
print(len(l2))
l3 = test_data['Uniprot'].values.tolist()
print(len(l3))
l4 = test_data['Int_uniprot'].values.tolist()
print(len(l4))

list3 = l1+l2+l3+l4
print(len(list3))
list3 = list(set(list3))
print(len(list3))


def read_pssm(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header and footer lines to only read the matrix section
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():  # Skip empty lines and lines not starting with a digit
            continue
        #print(line)
        row = line.split()[2:22]
        # Take only the PSSM values, skipping the position and amino acid columns
        #print(row)
        matrix.append(list(map(int, row)))
    # Convert the list to a NumPy array
    matrix = np.array(matrix)
    #print(matrix.shape)
    # Normalize the matrix values using the formula 1 / (1 + e^x)
    matrix = 1 / (1 + np.exp(-matrix))

    # Average the matrix along axis 0 to get a vector of 20 elements
    avg_vector = np.mean(matrix, axis=0)
    #print(avg_vector.shape)
    return avg_vector


print(read_pssm('C:/Users/kgera/Downloads/swissprot_pssms/swissprot_pssms/A8N2M6.pssm'))

pssm_path = 'C:/Users/kgera/Downloads/swissprot_pssms/swissprot_pssms'
arrays = []
succes = []
for uniprot in list3:
    try:
        name = f'{uniprot}.pssm'
        path = os.path.join(pssm_path, name)
        arrays.append(read_pssm(path))

        succes.append(uniprot)
    except:
        print('smth went wrong')
        print(uniprot)
        continue

print(arrays[0])
print(len(succes))

df = pd.DataFrame(succes)
df['matrices'] = arrays
print(df.shape)
df.to_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/all_pssms_22_42.csv')

