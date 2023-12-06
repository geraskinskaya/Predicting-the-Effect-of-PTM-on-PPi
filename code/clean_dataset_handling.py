import pandas as pd
import numpy as np
from pyfaidx import Fasta
from aaindex import aaindex1
from transformers import BertModel, BertTokenizer,BertConfig
import pandas as pd
import os
import umap




#Assuming you downloaded thsi repository, you can use 'your_path' to set your path to this directory,
#all further path references will be connected

your_path = ''

#load npz files with ProtBERT files with per-residue embeddings of proteins
f1 = os.join(your_path,'ProtBERT/file1/prepared_userds.npz' )
f2 = os.join(your_path, 'ProtBERT/file2/prepared_userds.npz')
f3 = os.join(your_path, 'ProtBERT/file3/prepared_userds.npz')
f4 = os.join(your_path, 'ProtBERT/protein_embeddings_nonhuman.npz')

file = np.load(f1)
file2 = np.load(f2)
file3 = np.load(f3)
file4 = np.load(f4)


#Function to read averaged (protein-level) embeddings from file, returning new dataset with this information
def create_new_dat(site, uniprot, Int_p, new_dataset, ptm, data, prots):
    your_path = 'code/content/all_means_final.csv'
    matrix = pd.read_csv(your_path)

    prot1 = matrix.loc[matrix[matrix.columns[1]] == uniprot].values.flatten().tolist()
    prot2 = matrix.loc[matrix[matrix.columns[1]] == Int_p].values.flatten().tolist()
    prot1 = prot1[2:1026]
    prot2 = prot2[2:1026]

    new_row = pd.Series(
        {'site': site, 'Uniprot': uniprot, 'protbert': prot1, 'Int_uniprot': Int_p,
         'int_protbert': prot2, 'PTM': ptm})
    new_dataset = pd.concat([new_dataset, new_row.to_frame().T], ignore_index=True)

    return new_dataset

# function to read files with transformed protein-level ProtBERT embeddings
def read_decomp(uniprot, int_p, decomp, matrix, site, ptm, new_dataset):
    prot1 = matrix.loc[matrix['un'] == uniprot].values.flatten().tolist()
    prot2 = matrix.loc[matrix['un'] == int_p].values.flatten().tolist()
    if decomp == 'umap':
        prot1 = prot1[1:9]
        prot2 = prot2[1:9]
    if decomp == 'tsne':
        prot1 = prot1[1:4]
        prot2 = prot2[1:4]
    if decomp == 'pca':
        prot1 = prot1[1:201]
        prot2 = prot2[1:201]
    if decomp == 'lbp':
        prot1 = prot1[1:257]
        prot2 = prot2[1:257]

    new_row = pd.Series(
        {'site': site, 'Uniprot': uniprot, 'protbert': prot1, 'Int_uniprot': int_p,
         'int_protbert': prot2, 'PTM': ptm})
    new_dataset = pd.concat([new_dataset, new_row.to_frame().T], ignore_index=True)

    return new_dataset

def create_protbert_window(Ub, lb, site, uniprot, wind):
    d_model = 1024
    sequence_length = len(str(protein_sequences[uniprot]))
    #encoding = positional_encoding(np.arange(sequence_length)[:, np.newaxis], d_model)
    # print(encoding.shape)
    try:
        matrice = file[uniprot]
    except KeyError:
        try:
            matrice = file2[uniprot]
        except KeyError:
            try:
                matrice = file3[uniprot]
            except KeyError:
                matrice = file4[uniprot]
                matrice = matrice[1::]

    if wind == True:
        st = (site-1) - lb
        end = (site-1) + Ub
        if (site-1) - lb <0:
            st = 0
        if (site-1)+Ub > sequence_length-1:
            end = sequence_length-1
        pre_out = np.array(matrice[st:end]) #+ 2*np.array(encoding[st:end])/3
        out = np.mean(pre_out, axis = 0)


    if wind == False:
        out = np.array(matrice[site-1])# + 2*np.array(encoding[site]))/3

    return out


def extract_id(header):
    return header.split('|')[1]



vals1 = aaindex1['CIDH920105'].values
vals2 = aaindex1['BHAR880101'].values
vals3 = aaindex1['CHAM820101'].values
vals4 = aaindex1['CHAM820102'].values
vals5 = aaindex1['CHOC760101'].values
vals6 = aaindex1['BIGC670101'].values
vals7 = aaindex1['CHAM810101'].values
vals8 = aaindex1['DAYM780201'].values
vals9 = aaindex1['GRAR740102'].values

interfaces = pd.read_csv('D:/ekat-test1/preds-UserDS_P.csv')
interfaces2 = pd.read_csv('D:/ekat-test2/preds-UserDS_P.csv')
interfaces3 = pd.read_csv('D:/ekat-test3/dnet-ppi/preds-UserDS_P.csv')
def read_wind_interface(site, ub, lb, uniprot):
    if uniprot in interfaces['prot_id'].values.tolist():
        pred = interfaces.loc[interfaces['prot_id'] == uniprot].values.tolist()
    elif uniprot in interfaces2['prot_id'].values.tolist():
        pred = interfaces2.loc[interfaces2['prot_id'] == uniprot].values.tolist()
    elif uniprot in interfaces3['prot_id'].values.tolist():
        print('here')
        pred = interfaces3.loc[interfaces3['prot_id'] == uniprot].values.tolist()
    else:
        W = -1
        return W

    print(len(pred))
    print(len(pred[0]))
    preds = pred[0][3]
    prds = [float(idx) for idx in preds.split(',')]
    print (len(prds))
    print(site)
    W = 0
    if prds[site-1] > 0.75:
        W = 1
    else:
        W = 0
    return W




STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
SAA_AAC = "ACDEFGHIKLMNPQRSTVWY"

protein_sequences = Fasta('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/uniprot_sprot.fasta', key_function=extract_id)

def aa_window_to_dataframe_row(aa_window, standard_aa='ACDEFGHIKLMNPQRSTVWY-'):
    aa_map = {aa: idx for idx, aa in enumerate(standard_aa)}
    row = [aa_map.get(aa, -1) for aa in aa_window]
    return row

def positional_encoding(position, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 0::2])
    return angle_rads


def threshold_function(x):
    return 1 if x >= 0 else 0



def read_data(fp, dat):
    data = pd.read_csv(fp)
    if dat == 'ptmint':
        data = data[data['Organism'] == 'Human']
        data = data[data['Uniprot'] != 'A9UF07']
        data = data[data['Uniprot'] != 'Q9Y5I7']
        data = data[data['Uniprot'] != 'Q16613']
        data = data[data['Uniprot'] != 'O23617']
#        data = data[data['Uniprot'] != 'P19456']
#        data.drop(columns=['PMID', 'Method', 'Gene', 'Disease', 'Int_gene', 'Co-localized', "Sequence window(-5,+5)",
#                           'Organism'], inplace=True)
        data.drop_duplicates(subset=['Uniprot', 'PTM', 'Site', 'AA', 'Int_uniprot', 'Effect'], inplace=True)
        print(data.shape)
        data = data[['Uniprot', 'PTM', 'Site', 'AA', 'Int_uniprot', 'Effect']]
        print(data.shape)
        data = data.replace({'Effect':{'Enhance':0, 'Inhibit':1, 'Induce': 2}})
        data = data.replace({'PTM':{'Ac':0, 'Glyco':1, 'Me':2, 'Phos':3, 'Sumo':4, 'Ub':5}})
        data['AA'] = data['AA'].replace({'T': 3, 'Y': 4, 'S': 2, 'K': 0, 'R': 1})
    if dat == 'bett':
        ptm = [3] * len(data)
        data['PTM'] = ptm
        new_column_order = ['Uniprot', 'PTM', 'Site', 'AA', 'Int_uniprot', 'Effect']
        data = data[new_column_order]
        data = data.replace({'Effect': {'Enhance': 0, 'Inhibit': 1}})
    data.reset_index(inplace = True)
    data = data[['Uniprot', 'PTM', 'Site', 'AA', 'Int_uniprot', 'Effect']]
    return data

fasta_file_path = 'C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/uniprot_sprot.fasta'
protein_sequences = Fasta(fasta_file_path, key_function=extract_id)

def create_window_and_seq(upper_boundary, lower_boundary, id, pos):
    seq = str(protein_sequences[id])
    #print(seq.shape)
    window = []
    for i in range(pos - lower_boundary, pos + upper_boundary):
        if i < 0 or i >= len(seq)-1:
            window.append('-')
        else:
            window.append(seq[i])
    window = ''.join(window)
    #print(window)
    return seq, window

def create_seq(id):
    seq = str(protein_sequences[id])
    return seq



def features(df, aac, aaindex, pssm, pos_enc, protbert, obc, int_wind, UBOUND, LBOUND, data, prots, decomp, umap_site):
    sequences = []
    windows = []
    sequences_Int = []
    for index, row in df.iterrows():
        seqs, winds = create_window_and_seq(UBOUND, LBOUND, row['Uniprot'], row['Site'])
        sequences.append(seqs)
        windows.append(winds)
        sequences_Int.append(create_seq(row['Int_uniprot']))

    df['Seq_p'] = sequences
    df['Window'] = windows
    df['Seq_Int'] = sequences_Int

    if int_wind == True:
        winds = []
        for index, row in df.iterrows():
            winds.append(read_wind_interface(row['Site'], UBOUND, LBOUND, row['Uniprot']))
        #j = range(9)
        #wind_df = pd.DataFrame(winds, columns = [f'int_w{a}' for a in j])
        #df = df.join(wind_df)
        df['Is_int'] = winds

    if pos_enc == True:
        positional_encodings = []
        d_model = 10
        for uniprot, site in zip(df['Uniprot'], df['Site']):
            #print(uniprot)
            sequence_length = len(protein_sequences[uniprot])
            encoding = positional_encoding(np.arange(sequence_length)[:, np.newaxis], d_model)
            # print(encoding.shape)
            positional_encodings.append(encoding[site - 1])
        df['pos'] = positional_encodings
        poses  = range(1,11)
        pos_df= pd.DataFrame(df['pos'].tolist(),
                                      columns=[f'pos{aa}' for aa in poses])
        df = df.join(pos_df)
        df.drop(columns = ['pos'], inplace = True)
    if protbert == True:
        new_datas = pd.DataFrame(columns=['site', 'Uniprot', 'protbert', 'int_uniprot', 'PTM'])
        for index, row in df.iterrows():
            try:
                new_datas = create_new_dat(row['Site'], row['Uniprot'], row['Int_uniprot'],
                                           new_datas, row['PTM'], data, prots)
            except IndexError:
                continue
        l = range(1024)
        df['int_protbert'] = new_datas['int_protbert']
        df['protbert'] = new_datas['protbert']
        p1 = pd.DataFrame(df['protbert'].tolist(), columns=[f'prot_1_{aa}' for aa in l])
        p2 = pd.DataFrame(df['int_protbert'].tolist(), columns=[f'prot_2_{aa}' for aa in l])
        #umap_transformer_subset = umap.UMAP(n_neighbors=6, n_components=8, random_state=42)
        #u1 = umap_transformer_subset.fit_transform(p1)
        #u2 = umap_transformer_subset.fit_transform(p2)
        #j = range(8)
        #df1 = pd.DataFrame(u1, columns=[f'P1_{a}' for a in j])
        #df2 = pd.DataFrame(u1, columns=[f'P2_{a}' for a in j])
        df = df.join(p1)
        df = df.join(p2)
        df.drop(columns = ['int_protbert', 'protbert'], inplace = True)

    if decomp == 'umap':
        d = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/umaps.csv')
        new_datas = pd.DataFrame(columns=['site', 'Uniprot', 'protbert', 'int_uniprot', 'PTM'])
        for index, row in df.iterrows():
            try:
                new_datas = read_decomp(row['Uniprot'], row['Int_uniprot'],
                                           decomp,d, row['Site'], row['PTM'], new_datas)
            except IndexError:
                continue
        l = range(8)
        df['int_protbert'] = new_datas['int_protbert']
        df['protbert'] = new_datas['protbert']
        p1 = pd.DataFrame(df['protbert'].tolist(), columns=[f'prot_1_{aa}' for aa in l])
        p2 = pd.DataFrame(df['int_protbert'].tolist(), columns=[f'prot_2_{aa}' for aa in l])
        print(p1.shape)
        print(p2.shape)
        df = df.join(p1)
        df = df.join(p2)
        df.drop(columns=['int_protbert', 'protbert'], inplace=True)
    if decomp == 'pca':
        d = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/Pca_composed.csv')
        new_datas = pd.DataFrame(columns=['site', 'Uniprot', 'protbert', 'int_uniprot', 'PTM'])
        for index, row in df.iterrows():
            try:
                new_datas = read_decomp(row['Uniprot'], row['Int_uniprot'],
                                        decomp, d, row['Site'], row['PTM'], new_datas)
            except IndexError:
                continue
        l = range(200)
        df['int_protbert'] = new_datas['int_protbert']
        df['protbert'] = new_datas['protbert']
        p1 = pd.DataFrame(df['protbert'].tolist(), columns=[f'prot_1_{aa}' for aa in l])
        p2 = pd.DataFrame(df['int_protbert'].tolist(), columns=[f'prot_2_{aa}' for aa in l])
        df = df.join(p1)
        df = df.join(p2)
    if decomp == 'lbp':
        d = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/Pca_composed.csv')
        new_datas = pd.DataFrame(columns=['site', 'Uniprot', 'protbert', 'int_uniprot', 'PTM'])
        for index, row in df.iterrows():
            try:
                new_datas = read_decomp(row['Uniprot'], row['Int_uniprot'],
                                        decomp, d, row['Site'], row['PTM'], new_datas)
            except IndexError:
                continue
        l = range(256)
        df['int_protbert'] = new_datas['int_protbert']
        df['protbert'] = new_datas['protbert']
        p1 = pd.DataFrame(df['protbert'].tolist(), columns=[f'prot_1_{aa}' for aa in l])
        p2 = pd.DataFrame(df['int_protbert'].tolist(), columns=[f'prot_2_{aa}' for aa in l])
        df = df.join(p1)
        df = df.join(p2)
    if decomp == 'tsne':
        d = pd.read_csv('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/tsne.csv')
        new_datas = pd.DataFrame(columns=['site', 'Uniprot', 'protbert', 'int_uniprot', 'PTM'])
        for index, row in df.iterrows():
            try:
                new_datas = read_decomp(row['Uniprot'], row['Int_uniprot'],
                                        decomp, d, row['Site'], row['PTM'], new_datas)
            except IndexError:
                continue
        l = range(3)
        df['int_protbert'] = new_datas['int_protbert']
        df['protbert'] = new_datas['protbert']
        p1 = pd.DataFrame(df['protbert'].tolist(), columns=[f'prot_1_{aa}' for aa in l])
        p2 = pd.DataFrame(df['int_protbert'].tolist(), columns=[f'prot_2_{aa}' for aa in l])
        df = df.join(p1)
        df = df.join(p2)
        df.drop(columns = ['protbert', 'int_protbert'], inplace = True)


    if obc == True:
        obcs = []
        for index, row in df.iterrows():
            obcs.append(aa_window_to_dataframe_row(row['Window']))
            #print(aa_window_to_dataframe_row(row['Window']))
        len_w = UBOUND+LBOUND
        dummy = range(len_w)
        obc_df = pd.DataFrame(obcs, columns = [f'W_{a}' for a in dummy])
        #print(obcs)
        df = df.join(obc_df)

    # 'protbert', 'int_protbert'
    if umap_site == 'wind':
        repr = []
        for index, row in df.iterrows():
            repr.append(create_protbert_window(UBOUND, LBOUND, row['Site'], row['Uniprot'], wind = True))
        j = range(1024)
        repr_df = pd.DataFrame(repr, columns = [f'wind_{a}' for a in j])
        #std_scaler = StandardScaler()
        #scaled_df = std_scaler.fit_transform(repr_df)
        #pca = PCA(n_components=200)
        #umap_embedding_subset = pca.fit_transform(scaled_df)
        #j = range(200)
        #df1 = pd.DataFrame(umap_embedding_subset, columns=[f'wind_prot_{a}' for a in j])
        #df = df.join(df1)
        df = df.join(repr_df)
    if umap_site == 'site':
        repr = []
        for index, row in df.iterrows():
            repr.append(create_protbert_window(UBOUND, LBOUND, row['Site'], row['Uniprot'], wind=False))
        j = range(1024)
        repr_df = pd.DataFrame(repr)
        umap_transformer_subset = umap.UMAP(n_neighbors=6, n_components=8, random_state=42)
        umap_embedding_subset = umap_transformer_subset.fit_transform(repr_df)
        j = range(8)
        df1 = pd.DataFrame(umap_embedding_subset, columns=[f'site_prot_{a}' for a in j])
        df = df.join(df1)
        df = df.join(repr_df)
    df.drop(columns=['Window', 'Seq_p', 'Seq_Int', 'Uniprot', 'Int_uniprot', 'AA'], inplace=True)
    return df

ptmint = read_data('C:/Users/kgera/PycharmProjects/data_preparation_thesis/content/Correct_filtered.csv', 'ptmint')
print('hey')
print(ptmint.shape)
PTM_new = features(ptmint, aac = False, aaindex = False, pssm = False, pos_enc = False, protbert = True, obc = False, int_wind = True, prots = 'none', UBOUND = 16, LBOUND = 15, data = 'ptmint', decomp = 'none', umap_site= 'wind')
print(PTM_new.shape)
betts = read_data('C:/Users/kgera/PycharmProjects/data_preparation_thesis/Bett.csv', 'bett')
betts_new = features(betts, aac = False, aaindex = False, pssm = False, pos_enc = False, protbert = True, obc = False, int_wind = True, prots = 'none', UBOUND = 16, LBOUND = 15, data = 'betts', decomp = 'none', umap_site= 'wind')
PTM_new.to_csv('Hum_ptmint_prot_el.csv')
betts_new.to_csv('Bett_FINAL_prot_el.csv')