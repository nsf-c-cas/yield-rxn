#!/usr/bin/env python3

"""
This script contains utility functions used in 
04_load_model.ipynb

"""
import os
import sys
import json
import warnings
import argparse
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

from IPython.display import SVG

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import DrawingOptions


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch._utils import _accumulate
from torch.utils.data import Dataset,Subset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,r2_score

###########################################
#Generate activations
###########################################
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []


        
def get_model_outputs(model,dataloader,data_smiles):
    data_dict=defaultdict(lambda: defaultdict())
    
    correct_yields=[]
    pred_yields=[]
    
    for i, data in enumerate(dataloader):
        rxn_id= data['id'].item()
        rxn_info = data_smiles[ data_smiles.id==int(rxn_id) ]
        yield_label = round(data['yield_label'].item(),2)
        rxn_smiles ='.'.join(rxn_info[['reactant_smiles','solvent_smiles','base_smiles']].values[0])

        ############### Define hook
        save_output = SaveOutput()
        hook_handles = [];j=0
        for layer in model.modules():
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
            j+=1

        mask_neis = torch.unsqueeze(data['n_bonds'].unsqueeze(-1) > torch.arange(0, 15, dtype=torch.int32).view(1, 1, -1), -1)
        max_n_atoms = data['n_atoms'].max()
        mask_atoms = torch.unsqueeze(data['n_atoms'].unsqueeze(-1) > torch.arange(0, max_n_atoms, dtype=torch.int32).view(1, -1),-1)
        with torch.no_grad():
            model.eval()
            yield_scores = model.forward(data['atom_feats'], data['bond_feats'],data['atom_graph'], data['bond_graph'], 
                            data['n_bonds'],data['n_atoms'], data['binary_feats'], mask_neis, mask_atoms,
                          data['sparse_idx'],data['domain_feats'])

            predicted = round(yield_scores.item(),2)

            
            correct_yields.append(yield_label)
            pred_yields.append(predicted)

            gnn_weights= save_output.outputs[9].detach().to('cpu').squeeze(0).squeeze(2).numpy()

            data_dict[rxn_id]['smiles'] = rxn_smiles 
            data_dict[rxn_id]['yield'] = yield_label
            data_dict[rxn_id]['pred_yield'] = predicted
            data_dict[rxn_id]['weights'] = gnn_weights
            
    r2score=r2_score(correct_yields,pred_yields)    
    
    
    return correct_yields,pred_yields,data_dict,r2score

###########################################
#Generate basic stats for predictions
###########################################

def get_basic_stats(data_dict,model_res,r2score, train_test ):
    "\nGenerating basic stats!"
    output_file = os.path.join(model_res,'summary.txt')
    low_low=0;low_high= 0
    high_low=0;high_high=0

    for rxn_id,res in data_dict.items():
        pred, y = res['pred_yield'],res['yield']
        if y<0.7:
            if pred <0.7:
                low_low+=1
            else:
                low_high+=1
        else:
            if pred<0.7:
                high_low +=1
            else:
                high_high+=1

    lows= low_low+low_high
    highs = high_low+high_high
    
    with open(output_file,'a') as g:
        g.write(f"{train_test} set\n")
        g.write(f"R2 score: {r2score}\n")
        g.write(f"Num low-yield: {lows}\n")
        g.write(f"Num high-yield: {highs}\n")
        g.write("Percentage of lows --> high: {:.2%}\n".format(low_high/lows,2))
        g.write("Percentage of lows --> low: {:.2%}\n".format(low_low/lows,2))
        g.write("Percentage of highs --> high: {:.2%}\n".format(high_high/highs,2))
        g.write("Percentage of highs --> low: {:.2%}\n\n\n".format(high_low/highs,2))
    g.close()    
    
    print(f"{train_test} set")
    print(f"R2 score: {r2score}")
    print(f"Num low-yield: {lows}")
    print(f"Num high-yield: {highs}")
    
    print("Percentage of lows --> high: {:.2%}".format(low_high/lows,2))
    print("Percentage of lows --> low: {:.2%}".format(low_low/lows,2))
    print("Percentage of highs --> high: {:.2%}".format(high_high/highs,2))
    print("Percentage of highs --> low: {:.2%}".format(high_low/highs,2))
    
###########################################
#get feature activation
###########################################

    
def get_domain_weights(domain_weights,domain_feature_names, model_res ):
    feat_names_weights= []
    output_file = os.path.join(model_res,'domain_weights.txt')
    
    for i in range(len(domain_feature_names)):
        feat_names_weights.append((domain_weights[0,i].item(),domain_feature_names[i]))
    sorted_feat_names_weights= sorted(feat_names_weights, key=lambda x: abs(x[0]),reverse=True)
    
    with open(output_file,'w') as g:
        for w,feat in sorted_feat_names_weights:
            g.write('\t'.join(map(str,[feat,round(w,3)]))+'\n')
    g.close()
    print(f"Domain weights written to {output_file}")
    return sorted_feat_names_weights    

###########################################
#plot activations
###########################################

def plot_activation_all_data(data_dict,train_test,model_res):
    print("\nGenerating activation figures...")
    for rxn_id,res in data_dict.items():
        smiles, weights = res['smiles'],res['weights']
        pred, y = round(res['pred_yield'],2),round(res['yield'],2)

        if abs(pred-y)>0.3:
            good_bad=train_test+'/bad/'
        else:
            good_bad=train_test+'/good/'

        mol = Chem.MolFromSmiles(smiles)
        bond_map = get_bond_map(mol)
        plot_activations(rxn_id,weights,mol,bond_map,y,pred ,model_res,good_bad)
    print('Done!')
        
def plot_activations(sample_num,gnn_weights,mol,bond_map,actual,predicted,model_res,good_bad):
    
    threshold= round(np.quantile(np.abs(gnn_weights),0.99),2)
    most_activated_atoms, most_activated_bonds = get_top_activated(gnn_weights, threshold,bond_map)
    fig_path=os.path.join(model_res,'vis',good_bad)
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    image_path=fig_path+ ''.join(map(str, [sample_num,'_Y_',actual,'_P_',predicted,'.png']))
    SVG(moltosvg(mol,most_activated_atoms,image_path,molSize=(600,600)))



def get_bond_map(mol):
    bond_map=defaultdict()
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_map[(a1,a2)] = bond.GetIdx()
        bond_map[(a2,a1)] = bond.GetIdx()
        #print(a1,a2,"bond: ",bond.GetIdx())
    return bond_map

    
    
def get_top_activated(gnn_weights, threshold,bond_map):
    most_activated_atoms=set()
    most_activated_bonds=set()
    for i in range(gnn_weights.shape[0]):
        for j in range(gnn_weights.shape[1]):
            if abs(gnn_weights[i][j])>threshold:
                most_activated_atoms.add(i)
                most_activated_atoms.add(j)
                #if (i,j) in bond_map:
                #most_activated_bonds.add(bond_map[(i,j)])
                    
    return most_activated_atoms, most_activated_bonds

 
def moltosvg(mol,highlightAtoms,output,molSize=(600,600),kekulize=True):
    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3.0

    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,highlightAtoms)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    d2d = Draw.MolDraw2DCairo(molSize[0],molSize[1])
    d2d.DrawMolecule(mc,highlightAtoms)
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()

    # save png to file
    with open(output, 'wb') as png_file:
        png_file.write(png_data)
    return svg.replace('svg:','')


###########################################
#Write train/test instances with pred/actual yield
###########################################

def write_model_preds(model_preds_fn,data_dict):
    print("\nWriting actual and predicted yield values for each sample...")
    with open(model_preds_fn,'w') as f:
        f.write(','.join(map(str,['id,yield,pred_yield,smiles\n'])))
        for rxn_id,res in data_dict.items():
            f.write(','.join(map(str,[rxn_id,res['yield'],res['pred_yield'],res['smiles']])) +'\n')
        
    f.close()    
    print('Done!')