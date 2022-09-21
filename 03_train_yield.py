#!/usr/bin/env python3
"""
This script loads the following:
-- data/<data_name>/<data_name>.json,
-- data/<data_name>/rf_results/selected_feats.txt,
-- <data_name>_raw_<use_rdkit>.csv,
-- and <data_name>_train_test_idxs.pickle
to split the data and train the GNN model on a 
data split. It outputs the R2 score and logs to 
-- output/<model_name>/
"""
import os
import sys
import json
import warnings
import argparse
import logging
import pickle
import pandas as pd
from collections import defaultdict

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import torch
from torch import randperm
from torch.utils.data import DataLoader
from torch._utils import _accumulate
from torch.utils.data import Dataset,Subset

from rxntorch.containers.reaction import Rxn
from rxntorch.containers.dataset import RxnGraphDataset as RxnGD
from rxntorch.utils import collate_fn
from rxntorch.models.yield_network import YieldNet, YieldTrainer
from sklearn.preprocessing import StandardScaler

#warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--dataset_path", type=str, default='./data/', help="train dataset")
parser.add_argument("-dn", "--dataset_name", required=True, type=str, help="dataset name. Options: az (AstraZeneca),dy (Doyle),su (Suzuki)")
parser.add_argument("-op", "--output_path", type=str, default='./output/', help="saved model path")
parser.add_argument("-mv", "--model_version", type=str, default='5.3.1-learn-w', help="Choose the model version")
parser.add_argument("-sn", "--split_set_num", type=int, default=1, help="Choose one split set for train and test. Options: 1-10")

parser.add_argument("-dr", "--dropout_rate", type=float, default=0.04, help="Ratio of samples to reserve for valid data")
parser.add_argument("-b", "--batch_size", type=int, default=40, help="number of batch_size")
parser.add_argument("-tb", "--test_batch_size", type=int, default=None, help="batch size for evaluation")
parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("-hs", "--hidden", type=int, default=200, help="hidden size of model layers")
parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")

parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of the optimizer")
parser.add_argument("-lrd", "--lr_decay", type=float, default=0.5, help="Decay factor for reducing the learning rate")
parser.add_argument("-lrs", "--lr_steps", type=int, default=10000,help="Number of steps between learning rate decay")

parser.add_argument("-awd","--adam_weight_decay", type=float, default=0.0, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

parser.add_argument("-gc", "--grad_clip", type=float, default=None, help="value for gradient clipping")
parser.add_argument("-pw", "--pos_weight", type=float, default=None, help="Weights positive samples for imbalance")

parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")
parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--cuda_devices", type=int, nargs='*', default=None, help="CUDA device ids")

parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("-ud","--use_domain", type=str, required=True, help="use domain features or not. options: rdkit: combination od rdkit feature and bozhao features. no_rdkit: only bozhao features. no_domain: neither.")
parser.add_argument("-mb","--max_nbonds", type=int, default=7, help="maximum number of bonds for binary features")
parser.add_argument("-ma","--max_natoms", type=int, default=200, help="maximum number of atoms for binary features")
parser.add_argument("--abs", type=str, default='abs', help="Take the average over aboslute/no absolute/sigmoid/relu value of predicted yield")



args = parser.parse_args()


#torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed )
torch.cuda.manual_seed_all(args.seed )


#input specs
data_type=args.dataset_name

#if use_domain=no_domain, just load either rdkit or no_rdkit .csv file and the
#set domain features to 0.
ext= '_'+args.use_domain if 'rdkit' in args.use_domain else '_no_rdkit' 
data_path = os.path.join(args.dataset_path,data_type)
processed_path = os.path.join(data_path,'processed-0')

input_split_idx_file = os.path.join(processed_path,'train_test_idxs.pickle')
processed_data_file = os.path.join(processed_path,''.join([data_type, ext,'.csv']))
selected_features_fn = os.path.join(data_path,'processed-0','rf_results','selected_feats.txt')


#output specs
# Saves model scores and the model itself in output_path
gc= 'gc' if args.grad_clip else ''
Abs=args.abs 

#model_name = '-'.join(map(str,[args.output_name, gc, args.use_domain, Abs, 'set',args.split_set_num, args.hidden, args.layers, args.epochs, args.lr, args.lr_decay, args.lr_steps,args.batch_size]))

#output_path= os.path.join(args.output_path ,model_name)
model_dir=  os.path.join('output',args.model_version)
output_name= data_type+'_'+args.model_version

model_name = '-'.join(map(str,[output_name, gc, args.use_domain, Abs, 'set',args.split_set_num,
                               args.hidden, args.layers, args.epochs, args.lr, args.lr_decay, 
                               args.lr_steps,args.batch_size]))

output_path= os.path.join(model_dir,model_name)
if not os.path.exists(output_path):
    os.mkdir(output_path)
#else:
    #logging.info(f"Model path created at : {output_path}")


logfile = '.'.join((output_name, "log"))
logpath = os.path.join(output_path, "logfile")
logging.basicConfig(level=logging.INFO, style='{', format="{asctime:s}: {message:s}",
                    datefmt="%m/%d/%y %H:%M:%S", handlers=(
                    logging.FileHandler(logpath), logging.StreamHandler()))

################################################################
#load_train_test sets
#################################################################
split_set_num= args.split_set_num

with open(input_split_idx_file, 'rb') as handle:
    idx_dict = pickle.load(handle)
    
selected_features = open(selected_features_fn,'r').readlines()[0].split(',')

logging.info("Loading Dataset in {dataset}".format( dataset=processed_data_file))
logging.info("Using the split set number {split}".format( split=split_set_num))


df=pd.read_csv(processed_data_file,index_col=0)
train_set= df.iloc[idx_dict['train_idx'][split_set_num]]
test_set = df.iloc[idx_dict['test_idx'][split_set_num]]

smiles_feature_names = ["id","yield","reactant_smiles","solvent_smiles","base_smiles","product_smiles"]
#domain_feature_names = ["yield"]+[f for f in df.columns if f not in smiles_feature_names]
domain_feature_names = [f for f in df.columns if f not in smiles_feature_names]


#apply feature selection
logging.info("Number of all available features: {num}".format(num=len(domain_feature_names)))
if args.use_domain=='rdkit':
    domain_feature_names = [f for f in domain_feature_names if f in selected_features]
    logging.info("Selecting features...")
    logging.info("Number of features after feature selection: {num}".format(num=len(domain_feature_names)))
else:
    logging.info("Not running feature selection!")


train_set_domain = train_set[domain_feature_names]
test_set_domain = test_set[domain_feature_names]
train_set_smiles = train_set[smiles_feature_names]
test_set_smiles = test_set[smiles_feature_names]

scaler = StandardScaler()

train_set_domain_scaled = pd.DataFrame(scaler.fit_transform(train_set_domain),columns = domain_feature_names)
test_set_domain_scaled = pd.DataFrame(scaler.transform(test_set_domain),columns = domain_feature_names)

assert train_set_domain.shape[0]  == train_set_smiles.shape[0] == train_set_domain_scaled.shape[0]
assert test_set_domain.shape[0]  == test_set_smiles.shape[0] == test_set_domain_scaled.shape[0]


################################################################
#Build RxnGD and DataLoaders
################################################################
logging.info("{:-^80}".format("Dataset"))
#feeding train smiles to test and vice versa to make sure our encoding is consistent.
# no label information is used on test set here.
train_dataset = RxnGD(train_set_domain_scaled,train_set_smiles, test_set_smiles, args.max_nbonds, args.max_natoms, args.use_domain)
test_dataset = RxnGD(test_set_domain_scaled, test_set_smiles, train_set_smiles, args.max_nbonds, args.max_natoms, args.use_domain)

                   
sample = train_dataset[3]
afeats_size, bfeats_size,  dmfeats_size = (sample["atom_feats"].shape[-1], sample["bond_feats"].shape[-1],
                                           sample['domain_feats'].shape[-1])





logging.info("{:d} samples for training ,{:d} samples for testing".format(train_set.shape[0], test_set.shape[0]))
logging.info("{:-^80}".format("Data loaders"))
logging.info("Batch size: {:d}  Workers: {:d}  Shuffle per epoch: {}".format(args.batch_size, args.num_workers, True))
logging.info("Drop incomplete batches: {}".format(True))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

test_batch_size = args.test_batch_size if args.test_batch_size is not None else args.batch_size
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=args.num_workers, 
                             collate_fn=collate_fn,drop_last=True)


logging.info("{:-^80}".format("Model"))
logging.info("Graph convolution layers: {}  Hidden size: {}".format(
    args.layers, args.hidden, args.batch_size, args.epochs))

################################################################
#Build RxnNet and RxnTrainer
################################################################

net = YieldNet(depth=args.layers, dropout= args.dropout_rate, afeats_size=afeats_size, bfeats_size=bfeats_size, hidden_size=args.hidden, 
                          dmfeats_size=dmfeats_size, max_nbonds=args.max_nbonds, use_domain=args.use_domain, abs_score=args.abs, max_natoms=args.max_natoms)
                          
logging.info("Total Parameters: {:,d}".format(sum([p.nelement() for p in net.parameters()])))

logging.info("{:-^80}".format("Trainer"))
logging.info("Optimizer: {}  Beta1: {}  Beta2: {}".format("Adam", args.adam_beta1, args.adam_beta2))
logging.info("Learning rate: {}  Learning rate decay: {}  Steps between updates: {}".format(
    args.lr, args.lr_decay, args.lr_steps))
logging.info("Weight decay: {} , Dropout Rate: {}, Gradient clipping: {}  Positive sample weighting: {}".format(
    args.adam_weight_decay, args.dropout_rate, args.grad_clip, args.pos_weight))

trainer = YieldTrainer(net, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                     with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                     grad_clip=args.grad_clip, pos_weight=args.pos_weight, lr_decay=args.lr_decay,
                     lr_steps=args.lr_steps, max_nbonds=args.max_nbonds)

################################################################
#Train
################################################################

train_r2 = open(output_path+'/train_scores.txt', 'a')
train_l = open(output_path+'/train_rmse.txt', 'a')
train_e = open(output_path+'/train_mae.txt', 'a')


test_r2 = open(output_path+'/test_scores.txt', 'a')
test_l = open(output_path+'/test_rmse.txt', 'a')
test_e = open(output_path+'/test_mae.txt', 'a')


#valid_r2 = open(output_path+'valid_scores.txt', 'a')
#valid_l = open(output_path+'valid_loss.txt', 'a')


w1_fn = open(output_path+'/weights_1.txt', 'a')
w2_fn = open(output_path+'/weights_2.txt', 'a')

max_score=0
for epoch in range(args.epochs):
    r2_train, train_loss,train_mae, w1,w2 = trainer.train_epoch(epoch, train_dataloader)
    r2_test ,test_loss,test_mae = trainer.test_epoch(epoch, test_dataloader)
    #r2_valid ,valid_loss = trainer.valid_epoch(epoch, valid_dataloader)
    
    train_r2.write(str(float(r2_train))+',')
    test_r2.write(str(float(r2_test))+',')
   
    
    train_l.write(str(float(train_loss))+',')
    test_l.write(str(float(test_loss))+',')
    
    train_e.write(str(float(train_mae))+',')
    test_e.write(str(float(test_mae))+',')

    w1_fn.write(str(float(w1))+',')
    w2_fn.write(str(float(w2))+',')
        
    if r2_test>max_score:
        trainer.save(epoch, model_name, model_dir)
        max_score=r2_test
    
                
train_r2.close();test_r2.close();#valid_r2.close()
train_l.close();test_l.close();#valid_l.close()
train_e.close();test_e.close();#valid_l.close()
w1_fn.close();w2_fn.close()