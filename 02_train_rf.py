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
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import mean_squared_error as s_rmse
from sklearn.metrics import mean_absolute_error as s_mae


from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-dn", "--dataset_name", type=str, default="dy",required=True, help="dataset name. Options: az (AstraZeneca),dy (Doyle),su (Suzuki)")
parser.add_argument("-dp","--dataset_path", type=str, default='./data/', help="dataset name")
parser.add_argument("-rdkit", "--use_rdkit_feats", default='rdkit', type=str, help="Use rdkit discriptors or not")
parser.add_argument("-od", "--output_dir", default='rf_results', type=str, help="Output dir for writing features and RF scores")
parser.add_argument("-th", "--feat_threshold",  type=float, default=0.0001, help="Threshold for feature importance to discard features")
parser.add_argument("-ne", "--n_estimators",  type=float, default=1000, help="Number of trees in RF model")
parser.add_argument("-md", "--max_depth",  type=float, default=10, help="Max depth in RF trees")
parser.add_argument("-rs", "--random_state",  type=int, default=0, help="Random state for RF model")
parser.add_argument("-plt", "--plot_yield_dist",  type=bool, default=False, help="Plot the yield distribution")
args = parser.parse_args()



data_type=args.dataset_name
use_rdkit_features= args.use_rdkit_feats
ext = '_'+ use_rdkit_features
processed= 'processed-0'#+str(args.random_state)
#inputs
processed_path = os.path.join(args.dataset_path,data_type,processed)

input_data_file = os.path.join(processed_path,''.join([data_type,ext ,'.csv']))
input_split_idx_file = os.path.join(processed_path,'train_test_idxs.pickle')


#outputs
output_path = os.path.join(args.dataset_path,data_type,processed,args.output_dir)
if not os.path.exists(output_path):
    os.mkdir(output_path)
features_fn= os.path.join(output_path,'selected_feats.txt')
r2_fn= os.path.join(output_path,'rf_results_r2'+ ext+'.csv')
mae_fn = os.path.join(output_path,'rf_results_mae'+ ext+'.csv')
rmse_fn = os.path.join(output_path,'rf_results_rmse'+ ext+'.csv')

print("\n\nReading data from: ",input_data_file)
print("Using rdkit features!") if use_rdkit_features=='rdkit' else print("Not using rdkit features!")


df=pd.read_csv(input_data_file,index_col=0)
smiles_features = ["reactant_smiles","solvent_smiles","base_smiles","product_smiles"]
df.drop(smiles_features, axis=1,inplace=True)
print(f"Raw data frame shape: {df.shape}")
if args.plot_yield_dist:
    print(f"Plotting yield distibution:")
    df['yield'].plot(kind='hist',bins=12)
    

def split_scale_data(df,split_set_num,idx_dict,label_name):
    """
    split the raw data into train and test using
    pre-writtten indexes. Then standardize the train
    and test set.
    """

    train_set = df.iloc[idx_dict['train_idx'][split_set_num]]
    test_set = df.iloc[idx_dict['test_idx'][split_set_num]]


    y_train, y_test = train_set.pop(label_name), test_set.pop(label_name)
    train_set.pop('id'), test_set.pop('id')
    x_train, x_test = train_set, test_set

    scaler = StandardScaler()

    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),columns = x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns = x_test.columns)
    return x_train_scaled, x_test_scaled , y_train, y_test



def get_sorted_feat_importances(feat_names,feat_importances):
    """
    sort the feature names based on RF feature importances
    and return the sorted feat names as well as pair:
    (feat_name, score)
    """
    sorted_idx = (-feat_importances).argsort()#[:n]

    sorted_feat_names = [feat_names[i] for i in sorted_idx]
    sorted_feat_importances =feat_importances[sorted_idx]
    final_feat_importances =list( zip(sorted_feat_names,sorted_feat_importances))

    return sorted_feat_names,final_feat_importances

def get_sorted_feat_importances(feat_names,feat_importances):
    """
    sort the feature names based on RF feature importances
    and return the sorted feat names as well as pair:
    (feat_name, score)
    """
    sorted_idx = (-feat_importances).argsort()#[:n]

    sorted_feat_names = [feat_names[i] for i in sorted_idx]
    sorted_feat_importances =feat_importances[sorted_idx]
    final_feat_importances =list( zip(sorted_feat_names,sorted_feat_importances))

    return sorted_feat_names,final_feat_importances



selected_features=set()
result_dict=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
with open(input_split_idx_file, 'rb') as handle:
    idx_dict = pickle.load(handle)

print()
for split_set_num in range(1,len(idx_dict['train_idx'])+1):
#for split_set_num in range(1,3):

    
    result_dict['r2'][split_set_num]['model_num']=split_set_num 
    result_dict['mae'][split_set_num]['model_num']=split_set_num 
    result_dict['rmse'][split_set_num]['model_num']=split_set_num 
    
    x_train_scaled, x_test_scaled , y_train, y_test= split_scale_data(df,split_set_num,idx_dict,'yield')
    rf = RandomForestRegressor(n_estimators = args.n_estimators, random_state = args.random_state ,max_depth=args.max_depth)
    y_test ,y_train = y_test/100 ,y_train/100

    rf.fit(x_train_scaled ,y_train)
    
    
    feat_names = list(x_train_scaled.columns) ##get the top features
    sorted_feat_names, sorted_feat_importances = get_sorted_feat_importances(feat_names,rf.feature_importances_)    
    for feat,score in sorted_feat_importances:
        result_dict['r2'][split_set_num][feat]=score
        result_dict['mae'][split_set_num][feat]=score
        result_dict['rmse'][split_set_num][feat]=score
        if score >= args.feat_threshold:
            selected_features.add(feat)

    y_pred_train = rf.predict(x_train_scaled) 
    y_pred_test = rf.predict(x_test_scaled)
    
    train_r2 = round(100*r2_score(y_train,y_pred_train),3)
    test_r2 = round(100*r2_score(y_test,y_pred_test ),3)
    
    result_dict['r2'][split_set_num]['train'] = round(r2_score(y_train,y_pred_train),3)
    result_dict['r2'][split_set_num]['test'] = round(r2_score(y_test,y_pred_test ),3)

    result_dict['mae'][split_set_num]['train'] = round(s_mae(y_train,y_pred_train),3)
    result_dict['mae'][split_set_num]['test'] = round(s_mae(y_test,y_pred_test ),3)

    result_dict['rmse'][split_set_num]['train'] = round(np.sqrt(s_rmse(y_train,y_pred_train)),3)
    result_dict['rmse'][split_set_num]['test'] = round(np.sqrt(s_rmse(y_test,y_pred_test )),3)


    print(f'\nModel number: {split_set_num}')
    print(f'Mean Absolute Train R2: ',100*result_dict['r2'][split_set_num]['train'])
    print(f'Mean Absolute Test R2: ',100*result_dict['r2'][split_set_num]['test'])
    
    print(f'\nMean Absolute Train RMSE: ',result_dict['rmse'][split_set_num]['train'])
    print(f'Mean Absolute Test RMSE: ',result_dict['rmse'][split_set_num]['test'])
    
    print(f'\nMean Absolute Train MAE: ',result_dict['mae'][split_set_num]['train'])
    print(f'Mean Absolute Test MAE: ',result_dict['mae'][split_set_num]['test'])


    
#Write the model summary results and selecyed features
dp_r2= pd.DataFrame.from_dict(result_dict['r2'], orient='index',columns=['model_num','train','test']+sorted_feat_names)
summary= dp_r2.describe().loc[['mean','std']]
dp_r2= dp_r2.append(summary)


dp_mae= pd.DataFrame.from_dict(result_dict['mae'], orient='index',columns=['model_num','train','test']+sorted_feat_names)
summary= dp_mae.describe().loc[['mean','std']]
dp_mae= dp_mae.append(summary)



dp_rmse= pd.DataFrame.from_dict(result_dict['rmse'], orient='index',columns=['model_num','train','test']+sorted_feat_names)
summary= dp_rmse.describe().loc[['mean','std']]
dp_rmse = dp_rmse.append(summary)


print(f"\nWriting selected features to: {features_fn}")
print(f"Writing RFs summary results to:\n{r2_fn},\n{mae_fn},\n{rmse_fn}")

dp_r2.to_csv(r2_fn)
dp_mae.to_csv(mae_fn)
dp_rmse.to_csv(rmse_fn)

if use_rdkit_features=='rdkit':
    with open(features_fn,'w') as f:
        f.write(','.join(map(str,selected_features)))