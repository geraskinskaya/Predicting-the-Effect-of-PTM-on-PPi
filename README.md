# Predicting-the-Effect-of-PTM-on-PPi
The source code for Master's Thesis project in Vrije Universiteit, Amsterdam. 

Abstract:
Post-translational modifications (PTMs) can considerably alter the structural or functional attributes of a protein, impacting, among others, protein-protein interactions (PPIs). This, in turn, can result in various pathogenic, degenerative and cancerous diseases. This research is aimed at developing an automated predictor for determining the effect of PTMs on PPIs, based on the PTMint database, a manually curated collection of experimentally verified effects of PTMs on PPIs. The LightGBM model was trained on a combination of ProtBERT embeddings of both interaction partner proteins, the Protein Site Window and a categorical feature representing whether or not PTM happened on the interface. The results show, that for the task of predicting the effect of Phosphorylation on PPIs, our approach performs better or just as well as previous attempts, while considerably decreasing computational costs. However, the performance of the model when applied to predicting the effect of multiple other PTM types on PPIs suggests that further research is needed to develop a way to represent PTMs.

Few notes for the user:

-repository contains compressed npz files with ProtBERT predictions, it is advised to decompress them manually, or modify the corresponding code.
-the code for filtering procedure is not presented, somehow the author performed it using Ubuntu command line, but the exact prompt got lost. Endlessly sorry for this, on the good side though - the filtered dataset is presented in the content folder.
