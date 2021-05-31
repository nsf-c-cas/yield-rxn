#!/usr/bin/env python3

"""
This script cleans and prcesses the json files
and generates a <data_name>_raw_<use_rdkit>.csv 
file which contains all possible features 
(depending on whther you use rdkit features or not). 
It also generates 10 sets of train-test indexes in
<data_name>_train_test_idxs.pickle
which are then used to split the data later for 
training and testing.
"""


import argparse
import logging
import os
import json
import math
import rdkit
import warnings
import pickle

import pandas as pd
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

from rxntorch.containers.reaction import Rxn
from sklearn.preprocessing import StandardScaler
from rdkit.ML.Descriptors import MoleculeDescriptors
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def clean(num):
    if num is None:
        return 0
    if num=='[]':
        return ''
    elif math.isnan(float(num)):
        return 0
    else:
        return float(num)
    
def build_rdkit_features(smile,comp_name):
    #print(smile,comp_name)
    descriptor_list=[x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)
    rdkit_feature_names=[comp_name+'_'+j for j in calculator.GetDescriptorNames()]
    
    
    mol=Chem.MolFromSmiles(smile)
    mol= Chem.AddHs(mol)
    rdkit_feature_values= calculator.CalcDescriptors(mol)
    rdkit_feats = dict(zip(rdkit_feature_names,rdkit_feature_values))       
    

    return rdkit_feats


parser = argparse.ArgumentParser()
parser.add_argument("-dn", "--dataset_name", type=str, default="dy",required=True, help="dataset name. Options: az (AstraZeneca),dy (Doyle),su (Suzuki)")
parser.add_argument("-dp","--dataset_path", type=str, default='./data/', help="dataset name")
parser.add_argument("-rdkit", "--use_rdkit_feats", required=True, type=int, help="Use rdkit discriptors or not")
parser.add_argument("-tr", "--test_ratio", type=float, required=True, default=0.3, help="test ratio for split")
parser.add_argument("-rs", "--random_state",  type=float, default=0, help="Random state for RF model")

args = parser.parse_args()



error_reaction_ids=set();
data_type=args.dataset_name
input_data= data_type+'_reactions_data.json'
use_rdkit_features= args.use_rdkit_feats
input_data_path = os.path.join(args.dataset_path,data_type,'raw',input_data)

ext = '_no_rdkit' if not use_rdkit_features else '_rdkit'
output_fn= ''.join([data_type,ext ,'.csv'])
output_path= os.path.join(args.dataset_path,data_type,'processed')
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_file= os.path.join(output_path,output_fn)


print("\n\nReading data from: ",input_data_path)
print("Using rdkit features!") if use_rdkit_features else print("Not using rdkit features!")


data_dict=defaultdict(lambda: defaultdict(float))

#get rdkit descriptors
descriptor_list=[x[0] for x in Descriptors._descList]
                
with open(input_data_path) as datafile:
    lines = json.load(datafile)

    for line in lines:
        solvent_key='solvent' if 'solvent' in line.keys() else 'Solvent' #case consitencies in data
        base_key= 'Base' if 'Base' in line.keys() else 'base' #case consitencies in data
        
        
        if data_type=='az':
            name= reaction_num = line["reaction_Num"]
            r_yield=line['yield']['yield']
        
        elif data_type in ['dy','su']:
            name= reaction_num = line["Id"]
            r_yield=line['yield']
        
        data_dict[name]["id"] = name
        reactants=line['reactants']   
        product_smiles = line.get("product",{}).get('smiles','')
        base_smile = line.get(base_key,{}).get('smiles','')
        
        # for Doyle data, solvent is the same for all reactions
        solvent_smile = 'CS(=O)C' if data_type=='dy' else line.get(solvent_key,[''])[0]
        if solvent_smile=='' or  solvent_smile==0:
            print(name)
            continue
        
        ########### other general reaction features 
        # temperature is different only in AZ data
        if data_type=='az': data_dict[name]['temperature'] = clean(line.get('temperature',0))
        elif data_type=='dy': data_dict[name]['temperature']=60.0
        elif data_type=='su': data_dict[name]['temperature']=100.0
            
        data_dict[name]['base_pka'] = clean(line.get(base_key,{}).get('Pka of Base-H',0))
        data_dict[name]['base_atom_cat'] = clean(line.get(base_key,{}).get('Atomic_number_Cation',0))   
        
        ########## other features in AZ data
        data_dict[name]['reaction_scale'] = clean(line.get('scale',0))
        data_dict[name]['base_amount'] = clean(line.get('base_amount',0))
        data_dict[name]['catalyst_amount'] = clean(line.get('catalyst_amount',0))
        data_dict[name]['reaction_volume'] = clean(line.get('volume',0))
        ##########
        
        #get solvent values
        solvent_vec= line.get(solvent_key,[])
        
        if solvent_vec !=[]:
            for i in range(1,len(solvent_vec)):
                data_dict[name]['solvent_'+str(i)]=float(solvent_vec[i])
        
        rxn = Rxn(name,reactants,solvent_smile,base_smile,r_yield)
        mol_reactants=rxn.mol_reactants

                
        for mol_idx in range(len(mol_reactants)):
            current_molecule = mol_reactants[mol_idx]
            category = current_molecule.category
            mol_smiles= current_molecule.smile
            
            #doing this helps with mapping reactions form Su to Dy and Az together
            #if category=='Boronic Acid': category='Amine' 
            
            #calculate rdkit features for reactants and integrate into data_dict
            if use_rdkit_features:
                rdkit_feats_curr_mol = build_rdkit_features(mol_smiles,category)
                for key, value in rdkit_feats_curr_mol.items():
                    data_dict[name][key] = value
            
            
            vib_modes = current_molecule.vib_modes

            atoms =current_molecule.atoms
            all_attributes=current_molecule.get_attributes().squeeze().tolist()
            weight,volume,surface_area,ovality,hardness,dipole_moment,electronegativity,HOMO,LUMO=all_attributes
            data_dict[name][category] = current_molecule.name
            data_dict[name][category +'_molecular_weight'] = round(float(weight),5)
            data_dict[name][category +'_molecular_volume'] = round(float(volume),5)
            data_dict[name][category +'_surface_area'] = round(float(surface_area),5)
            data_dict[name][category +'_ovality'] = round(float(ovality),5)
            data_dict[name][category +'_hardness'] = round(float(hardness),5)
            data_dict[name][category +'_dipole_moment'] = round(float(dipole_moment),5)
            data_dict[name][category +'_electronegativity'] = round(float(electronegativity),5)
            data_dict[name][category +'_E_HOMO'] = round(float(HOMO),5)
            data_dict[name][category +'_E_LOMO'] = round(float(LUMO),5)
            for n in range(len(vib_modes)):
                data_dict[name][category + '_V'+str(n)+'_frequency'] = round(float(vib_modes[n][0]),5)
                data_dict[name][category + '_V'+str(n)+'_intensity'] = round(float(vib_modes[n][1]),5)
            
            for atom in atoms:
                if 'H' not in atom['name']: #exculding hydrogen for now
                    if 'partial_charge' in atom:
                        data_dict[name][category+'_.'+atom['name']+'_electrostatic_charge']=round(float(clean(atom['partial_charge'])),5)
                                     
                    if 'nmr_shift' in atom:
                        data_dict[name][category+'_.'+atom['name']+'_NMR_shift']= round(float(clean(atom['nmr_shift'])),5)
                                                
            ######################### 
            #get rdkit fetures for base, solvent, product
            if use_rdkit_features:
                all_comps = [solvent_smile,base_smile,product_smiles]
                comp_map= {0:'solvent',1:'base',2:'product'}
                rdkit_feats_combined={}
                for i,smile in enumerate(all_comps):
                    if smile not in ['',' ',0]:
                        rdkit_feats_combined = build_rdkit_features(smile,comp_map[i])            

                        for key, value in rdkit_feats_combined.items():
                            data_dict[name][key] = value
                
            ###############################            
        if isfloat(r_yield):                
            data_dict[name]['yield'] = round(float(r_yield),5)      
        else:
            del data_dict[name]
            error_reaction_ids.add(name)
            
            
print(f"\nNumber of reactions with problematic yield: {len(error_reaction_ids)}")
print(f"Number of valid reactions in data dict: {len(data_dict)}")


#Convert data_dict to pandas dataframe
df_o=pd.DataFrame.from_dict(data_dict, orient='index')
df_o=df_o.reset_index()
categorical =list( df_o.columns[ ~( (df_o.dtypes.values == np.dtype('float64')) | (df_o.dtypes.values == np.dtype('int64')))])
zero_val_features = list(df_o.columns[(df_o == 0).all()])

print(f"Number of all features: {df_o.shape[1]}")
print(f"Number of catgorical features: {len(categorical)}")
print(f"Number of zero-value features: {len(zero_val_features)}")


to_drop=zero_val_features+categorical+['index']
print(f"\nDropping {len(to_drop)} zero-val and categorical features...")

df= df_o.copy()
df.drop(to_drop, axis=1, inplace=True)
curr_cols=list(df.columns)

if 'yield' in curr_cols: # put "yield" in the first column
    curr_cols.remove('yield')
new_cols=['yield']+curr_cols
print(f"Number of features after dropping: {len(new_cols)}")


df.fillna(0, inplace=True)
df=df[new_cols]
data_dict_transformed =df.transpose().to_dict(orient='list')

print(f"\nWriting csv file to: {output_file}")
df.to_csv(output_file)