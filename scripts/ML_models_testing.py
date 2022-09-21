#!/usr/bin/env python3

"""
This script loads the <data_name>_raw_<use_rdkit>.csv
file and <data_name>_train_test_idxs.pickle to split
the data and train a random forest on each data split.
It outputs the RF R2 score and the selcted features to
data/<data_name>/rf_results/rf_results.csv
data/<data_name>/rf_results/selected_feats.txt
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error as s_rmse
from sklearn.metrics import mean_absolute_error as s_mae
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-dn", "--dataset_name", type=str, default="su",
                    help="dataset name. Options: az (AstraZeneca),dy (Doyle),su (Suzuki)")
parser.add_argument("-dp", "--dataset_path", type=str, default='./data/', help="dataset name")
parser.add_argument("-rdkit", "--use_rdkit_feats", default='rdkit', type=str, help="Use rdkit discriptors or not")
parser.add_argument("-od", "--output_dir", default='rf_results', type=str,
                    help="Output dir for writing features and RF scores")
parser.add_argument("-ne", "--n_estimators", type=float, default=100, help="Number of trees in RF model")
parser.add_argument("-md", "--max_depth", type=float, default=10, help="Max depth in RF trees")
parser.add_argument("-rs", "--random_state", type=int, default=1, help="Random state for RF model")
parser.add_argument("-plt", "--plot_yield_dist", type=bool, default=False, help="Plot the yield distribution")
parser.add_argument("-cv", "--cv", type=int, default=5, help="Folds for cross validation")
parser.add_argument("-ft", "--fine_tuning", type=bool, default=True, help="Fine_tune the superparameters")
parser.add_argument("-sf", "--Shuffle", default=False, action='store_true', help="Shuffle the reaction and yield")
args = parser.parse_args()
Shuffle = args.Shuffle

data_type = args.dataset_name
use_rdkit_features = args.use_rdkit_feats
ext = '_' + use_rdkit_features
processed = 'processed-0'  # +str(args.random_state)
# inputs
processed_path = os.path.join(args.dataset_path, data_type, processed)

input_data_file = os.path.join(processed_path, ''.join([data_type, ext, '.csv']))
input_split_idx_file = os.path.join(processed_path, 'train_test_idxs.pickle')

# outputs
output_path = os.path.join(args.dataset_path, data_type, processed, args.output_dir)
if not os.path.exists(output_path):
    os.mkdir(output_path)

r2_fn = os.path.join(output_path, 'rf_results_r2' + ext + '.csv')
mae_fn = os.path.join(output_path, 'rf_results_mae' + ext + '.csv')
rmse_fn = os.path.join(output_path, 'rf_results_rmse' + ext + '.csv')
total_fn = os.path.join(output_path, 'rf_total' + ext + '.csv')


print("\n\nReading data from: ", input_data_file)
print("Using rdkit features!") if use_rdkit_features == 'rdkit' else print("Not using rdkit features!")

df = pd.read_csv(input_data_file, index_col=0)

smiles_features = ["reactant_smiles","solvent_smiles","base_smiles","product_smiles"]
df.drop(smiles_features, axis=1,inplace=True)

print(f"Raw data frame shape: {df.shape}")
if args.plot_yield_dist:
    print(f"Plotting yield distibution:")
    df['yield'].plot(kind='hist', bins=12)


def split_scale_data(df, split_set_num, idx_dict, label_name):
    """
    split the raw data into train and test using
    pre-writtten indexes. Then standardize the train
    and test set.
    """

    train_set = df.iloc[idx_dict['train_idx'][split_set_num]]
    test_set = df.iloc[idx_dict['test_idx'][split_set_num]]

    train_set.pop('id'), test_set.pop('id')
    y_train, y_test = train_set.pop(label_name), test_set.pop(label_name)
    x_train, x_test = train_set, test_set
    

    
    if args.Shuffle:
      y_train = shuffle(y_train, random_state=0)
      y_test = shuffle(y_test, random_state=0)
      print('shuffle finished')

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    return x_train_scaled, x_test_scaled, y_train, y_test


def get_sorted_feat_importances(feat_names, feat_importances):
    """
    sort the feature names based on RF feature importances
    and return the sorted feat names as well as pair:
    (feat_name, score)
    """
    sorted_idx = (feat_importances).argsort()  # [:n]

    sorted_feat_names = [feat_names[i] for i in sorted_idx]
    sorted_feat_importances = feat_importances[sorted_idx]
    final_feat_importances = list(zip(sorted_feat_names, sorted_feat_importances))

    return sorted_feat_names, final_feat_importances


selected_features = set()
result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
with open(input_split_idx_file, 'rb') as handle:
    idx_dict = pickle.load(handle)


distributions = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'])


if args.fine_tuning:
    print('start fine-tuning')
    
    x_train_scaled, x_test_scaled, y_train, y_test = split_scale_data(df, 1, idx_dict, 'yield')
    
    rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state, max_depth=args.max_depth)
    n_estimators = list(range(1,100))
    hyperpara = [{ 'n_estimators' : n_estimators}]
    rf = GridSearchCV(rf,hyperpara, cv=args.cv)
    
    lasso = Lasso(alpha=0.1)
    alphas = np.logspace(-4, -0.5, 40)
    tuned_parameters = [{'alpha': alphas}]

    lasso = GridSearchCV(lasso,tuned_parameters,cv=args.cv)
    
    lasso_cv = LassoCV(cv=args.cv ,random_state=0)
     
    KNN = KNeighborsRegressor()
    n_neighbors = list(range(1,30))
    hyperparameters = dict(n_neighbors=n_neighbors)
    KNN = GridSearchCV(KNN,hyperparameters,cv=args.cv)
    
    alphas = [1E-8,1e-7,1e-6,1e-5,1e-4,1e-3]
    hyper = [{'alpha':alphas}]
    NN = MLPRegressor(alpha=1e-6, hidden_layer_sizes=(10, 4), random_state=1)
    NN = RandomizedSearchCV(NN,hyper, cv = args.cv)
    
    y_test, y_train = y_test / 100, y_train / 100
    
    sh1 = rf.fit(x_train_scaled, y_train)
    print(sh1.best_estimator_)
    sh2 = lasso.fit(X=x_train_scaled, y=y_train)
    print(sh2.best_estimator_)
    sh4 = KNN.fit(x_train_scaled, y_train)
    print(sh4.best_estimator_)
    sh6 = NN.fit(X=x_train_scaled, y=y_train)
    print(sh6.best_estimator_)
    print('parameter fine-tuning done')


for split_set_num in range(1, len(idx_dict['train_idx']) + 1):
    # for split_set_num in range(1,3):

    result_dict['r2'][split_set_num]['model_num'] = split_set_num
    result_dict['mae'][split_set_num]['model_num'] = split_set_num
    result_dict['rmse'][split_set_num]['model_num'] = split_set_num

    x_train_scaled, x_test_scaled, y_train, y_test = split_scale_data(df, split_set_num, idx_dict, 'yield')
    y_test, y_train = y_test / 100, y_train / 100
    
    rf = sh1.best_estimator_
    lasso = sh2.best_estimator_
    lasso_cv = LassoCV(cv=args.cv,random_state=0)
    KNN = sh4.best_estimator_
    SVM = svm.SVR()
    NN = sh6.best_estimator_
    
    rf.fit(x_train_scaled, y_train)
    lasso.fit(X=x_train_scaled, y=y_train)
    lasso_cv.fit(x_train_scaled, y_train)
    KNN.fit(x_train_scaled, y_train)
    SVM.fit(x_train_scaled, y_train)
    NN.fit(X=x_train_scaled, y=y_train)
    
    
    rf_y_pred_train = rf.predict(x_train_scaled)
    rf_y_pred_test = rf.predict(x_test_scaled)
  

    lasso_y_pred_train = lasso.predict(x_train_scaled)
    lasso_y_pred_test = lasso.predict(x_test_scaled)

    lasso_cv_y_pred_train = lasso_cv.predict(x_train_scaled)
    lasso_cv_y_pred_test = lasso_cv.predict(x_test_scaled)


    KNN_y_pred_train = KNN.predict(x_train_scaled)
    KNN_y_pred_test = KNN.predict(x_test_scaled)

    SVM_y_pred_train = SVM.predict(x_train_scaled)
    SVM_y_pred_test = SVM.predict(x_test_scaled)

    NN_y_pred_train = NN.predict(x_train_scaled)
    NN_y_pred_test = NN.predict(x_test_scaled)
    
    
    train_r2 = round(r2_score(y_train, rf_y_pred_train), 3)
    test_r2 = round(r2_score(y_test, rf_y_pred_test), 3)
    
    result_dict['r2_rf'][split_set_num]['train'] = round(r2_score(y_train, rf_y_pred_train), 3)
    result_dict['r2_rf'][split_set_num]['test'] = round(r2_score(y_test, rf_y_pred_test), 3)

    result_dict['r2_lasso'][split_set_num]['train'] = round(r2_score(y_train, lasso_y_pred_train), 3)
    result_dict['r2_lasso'][split_set_num]['test'] = round(r2_score(y_test, lasso_y_pred_test), 3)

    result_dict['r2_lasso_cv'][split_set_num]['train'] = round(r2_score(y_train, lasso_cv_y_pred_train), 3)
    result_dict['r2_lasso_cv'][split_set_num]['test'] = round(r2_score(y_test, lasso_cv_y_pred_test), 3)

    result_dict['r2_svm'][split_set_num]['train'] = round(r2_score(y_train, SVM_y_pred_train), 3)
    result_dict['r2_svm'][split_set_num]['test'] = round(r2_score(y_test, SVM_y_pred_test), 3)

    result_dict['r2_nn'][split_set_num]['train'] = round(r2_score(y_train, NN_y_pred_train), 3)
    result_dict['r2_nn'][split_set_num]['test'] = round(r2_score(y_test, NN_y_pred_test), 3)

    result_dict['r2_knn'][split_set_num]['train'] = round(r2_score(y_train, KNN_y_pred_train), 3)
    result_dict['r2_knn'][split_set_num]['test'] = round(r2_score(y_test, KNN_y_pred_test), 3)
    
    result_dict['mae_rf'][split_set_num]['train'] = round(s_mae(y_train, rf_y_pred_train), 3)
    result_dict['mae_rf'][split_set_num]['test'] = round(s_mae(y_test, rf_y_pred_test), 3)

    result_dict['mae_lasso'][split_set_num]['train'] = round(s_mae(y_train, lasso_y_pred_train), 3)
    result_dict['mae_lasso'][split_set_num]['test'] = round(s_mae(y_test, lasso_y_pred_test), 3)

    result_dict['mae_lasso_cv'][split_set_num]['train'] = round(s_mae(y_train, lasso_cv_y_pred_train), 3)
    result_dict['mae_lasso_cv'][split_set_num]['test'] = round(s_mae(y_test, lasso_cv_y_pred_test), 3)

    result_dict['mae_svm'][split_set_num]['train'] = round(s_mae(y_train, SVM_y_pred_train), 3)
    result_dict['mae_svm'][split_set_num]['test'] = round(s_mae(y_test, SVM_y_pred_test), 3)

    result_dict['mae_nn'][split_set_num]['train'] = round(s_mae(y_train, NN_y_pred_train), 3)
    result_dict['mae_nn'][split_set_num]['test'] = round(s_mae(y_test, NN_y_pred_test), 3)

    result_dict['mae_knn'][split_set_num]['train'] = round(s_mae(y_train, KNN_y_pred_train), 3)
    result_dict['mae_knn'][split_set_num]['test'] = round(s_mae(y_test, KNN_y_pred_test), 3)
    

  
    print(f'\nModel number: {split_set_num}')
    print(f'Mean Absolute Train R2: ', train_r2)
    print(f'Mean Absolute Test R2: ', test_r2)

 
dp_r2_rf = pd.DataFrame.from_dict(result_dict['r2_rf'], orient='index',
                               columns=['model_num', 'train', 'test'] )

dp_r2_lasso = pd.DataFrame.from_dict(result_dict['r2_lasso'], orient='index',
                               columns=['model_num', 'train', 'test'])
                               
dp_r2_lasso_cv = pd.DataFrame.from_dict(result_dict['r2_lasso_cv'], orient='index',
                               columns=['model_num', 'train', 'test'] )

dp_r2_svm = pd.DataFrame.from_dict(result_dict['r2_svm'], orient='index',
                               columns=['model_num', 'train', 'test'])

dp_r2_nn = pd.DataFrame.from_dict(result_dict['r2_nn'], orient='index',
                               columns=['model_num', 'train', 'test'])

dp_r2_knn = pd.DataFrame.from_dict(result_dict['r2_knn'], orient='index',
                               columns=['model_num', 'train', 'test'])

dp_r2=pd.concat([dp_r2_rf,dp_r2_lasso,dp_r2_lasso_cv,dp_r2_svm,dp_r2_nn,dp_r2_knn] ,axis=1,join = 'inner' )


dp_mae_rf = pd.DataFrame.from_dict(result_dict['mae_rf'], orient='index',
                               columns=['model_num', 'train', 'test'] )

dp_mae_lasso = pd.DataFrame.from_dict(result_dict['mae_lasso'], orient='index',
                               columns=['model_num', 'train', 'test'])
                               
dp_mae_lasso_cv = pd.DataFrame.from_dict(result_dict['mae_lasso_cv'], orient='index',
                               columns=['model_num', 'train', 'test'] )

dp_mae_svm = pd.DataFrame.from_dict(result_dict['mae_svm'], orient='index',
                               columns=['model_num', 'train', 'test'])

dp_mae_nn = pd.DataFrame.from_dict(result_dict['mae_nn'], orient='index',
                               columns=['model_num', 'train', 'test'])

dp_mae_knn = pd.DataFrame.from_dict(result_dict['mae_knn'], orient='index',
                               columns=['model_num', 'train', 'test'])

dp_mae=pd.concat([dp_mae_rf,dp_mae_lasso,dp_mae_lasso_cv,dp_mae_svm,dp_mae_nn,dp_mae_knn] ,axis=1,join = 'inner' )

dp_r2.to_csv(r2_fn)
dp_mae.to_csv(mae_fn)




