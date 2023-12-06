from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import torch

import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

import torch
print(torch.cuda.is_available())


dirpath = os.path.dirname('C:/Users/kgera/PycharmProjects/data_preparation_thesis/')
filepath = os.path.join(dirpath, 'content/Train_WNEG.csv')
data = pd.read_csv(filepath)
filepath_test = os.path.join(dirpath, 'content/Test_pure_final.csv')
test_data = pd.read_csv(filepath_test)
data = data.replace({'Effect':{1:0, 2:1}})
test_data = test_data.replace({'Effect':{1:0, 2:1}})


#print(data['AA'].dtypes)
data.drop(columns = ['Organism', 'Uniprot', 'Int_uniprot'], inplace = True)
test_data.drop(columns = ['Organism', 'Uniprot', 'Int_uniprot'], inplace = True)

data['AA'] = data.AA.astype('category')
test_data['AA'] = test_data.AA.astype('category')
data['PTM'] = data.PTM.astype('category')
test_data['PTM'] = test_data.PTM.astype('category')
cat_columns = ['AA', 'PTM' ]

l = []
for val in data['Co-localized']:
  if val == '':
    l.append(0)
  else:
    l.append(1)
data['Co-loc'] = l

l = []
for val in test_data['Co-localized']:
  if val == '':
    l.append(0)
  else:
    l.append(1)
test_data['Co-loc'] = l

data.drop(['Co-localized'], axis = 1, inplace = True)
test_data.drop(['Co-localized'], axis = 1, inplace = True)

data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)
# Assuming 'data' is your DataFrame and 'Effect' is the column you want to predict
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
test_data = test_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_imb = data.drop(['Effect'], axis=1)
y_imb = data['Effect']
X_test = test_data.drop(['Effect'], axis=1)
y_test = test_data['Effect']


ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X_imb, y_imb)


# Split the data into training and validation sets
X_train_tab, X_val_tab, y_train_tab, y_val_tab = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tab = X_train_tab.values
y_train_tab = y_train_tab.values
X_val_tab = X_val_tab.values
y_val_tab = y_val_tab.values
X_test = X_test.values
y_test = y_test.values


param_grid = {
    'n_d': [8, 16, 32],
    'n_a': [8, 16, 32],
    'n_steps': [3, 5, 7],
    'gamma': [1.0, 1.3, 1.5],
    'lambda_sparse': [0, 0.0001, 0.001]
}

# Find the best hyperparameters by cross-validation
best_auc = 0
best_params = None

for params in ParameterGrid(param_grid):
    tabnet_model = TabNetClassifier(
        device_name='cuda',
        verbose=1,
        **params
    )
    tabnet_model.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_train_tab, y_train_tab), (X_val_tab, y_val_tab)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=100,
        patience=10,
        batch_size=256
    )
    val_auc = tabnet_model.best_cost
    if val_auc > best_auc:
        best_auc = val_auc
        best_params = params

print(f"Best AUC on validation set: {best_auc}")
print(f"Best hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
final_tabnet_model = TabNetClassifier(
    device_name='cuda',
    verbose=1,
    **best_params
)
final_tabnet_model.fit(
    X_train_tab, y_train_tab,
    max_epochs=1000
)

# Predict on the test set
y_pred_tab = final_tabnet_model.predict(X_test)
print(classification_report(y_test, y_pred_tab))
print(confusion_matrix(y_test, y_pred_tab))
print(roc_auc_score(y_test, y_pred_tab))

# Compute ROC curve and ROC area for each class
fpr_tab, tpr_tab, _ = roc_curve(y_test, y_pred_tab)
roc_auc_tab = auc(fpr_tab, tpr_tab)

# Plot
plt.figure()
plt.plot(fpr_tab, tpr_tab, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_tab)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('tabnet_roc_auc.png')
plt.close()
