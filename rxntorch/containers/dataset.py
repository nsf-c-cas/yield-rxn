from __future__ import print_function
from __future__ import division

import os
import re
import glob
import math
import json
import logging
import pandas as pd

import rdkit.Chem as Chem

from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from collections import defaultdict

from .reaction import Rxn


class RxnGraphDataset():
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.
    """    
    def __init__(self, df_domain, df_smiles, df_other_smiles, max_nbonds, max_natoms, use_domain):
        #super(RxnGraphDataset, self).__init__()
        #self.file_name = id
        self.df_domain= df_domain
        self.df_smiles= df_smiles
        self.df_other_smiles = df_other_smiles
        self.rxns = []
        self.degree_codec = LabelEncoder()
        self.symbol_codec = LabelEncoder()
        self.expl_val_codec = LabelEncoder()
        self.bond_type_codec = LabelEncoder()
        self.domain_feats=defaultdict(list)      
        self.max_nbonds = max_nbonds
        self.max_natoms = max_natoms
        self.max_n_all_atoms=0
        self.use_domain = use_domain
        self._init_dataset()
    
    def __len__(self):
        return len(self.rxns)
        
        
    def _init_dataset(self):
        
        symbols = set()
        degrees = set()
        explicit_valences = set()
        bond_types = set()
                
        self.domain_feat_shape = self.df_domain.iloc[0].shape
        for i in range(self.df_smiles.shape[0]):
            reaction_id =  self.df_smiles.iloc[i]["id"]
            reactant_smiles= self.df_smiles.iloc[i]["reactant_smiles"]
            solvent_smiles = self.df_smiles.iloc[i]["solvent_smiles"]
            base_smiles =  self.df_smiles.iloc[i]["base_smiles"]
            product_smiles =  self.df_smiles.iloc[i]["product_smiles"]
            rxn_yield= self.df_smiles.iloc[i]["yield"]
            
            if self.use_domain:
                #domain_features =  torch.from_numpy(df_domain.iloc[i].values).unsqueeze(0)
                
                domain_features = torch.tensor(list(self.df_domain.iloc[i].values),dtype=torch.float).unsqueeze(0)
                
            else:
                
                domain_features = torch.zeros((self.domain_feat_shape),dtype=torch.float).unsqueeze(0)
            self.domain_feats[reaction_id]= domain_features

            rxn = Rxn(reaction_id,reactant_smiles,solvent_smiles,base_smiles,product_smiles,rxn_yield)
            reactants=rxn.reactants

            n_atoms=Chem.MolFromSmiles(rxn.reactants_smile).GetNumAtoms()
            self.max_n_all_atoms = max(self.max_n_all_atoms, n_atoms)

            self.rxns.append(rxn)
            mol = Chem.MolFromSmiles(rxn.reactants_smile) #reactants +solvent +base

            for atom in mol.GetAtoms():   
                symbols.add(atom.GetSymbol())
                degrees.add(atom.GetDegree())
                explicit_valences.add(atom.GetExplicitValence())
            for bond in mol.GetBonds():
                bond_types.add(bond.GetBondType())
        
        logging.info("Loading the other dataset (not used in training):\n")
        for i in range(self.df_other_smiles.shape[0]):
            reactant_smiles= self.df_other_smiles.iloc[i]["reactant_smiles"]
            solvent_smiles = self.df_other_smiles.iloc[i]["solvent_smiles"]
            base_smiles =  self.df_other_smiles.iloc[i]["base_smiles"]

            all_reactant_smiles= '.'.join([reactant_smiles,solvent_smiles,base_smiles])  
            mol = Chem.MolFromSmiles(all_reactant_smiles)
            for atom in mol.GetAtoms():   
                symbols.add(atom.GetSymbol())
                degrees.add(atom.GetDegree())
                explicit_valences.add(atom.GetExplicitValence())
            for bond in mol.GetBonds():
                bond_types.add(bond.GetBondType())
                
        symbols.add("unknown")
        logging.info("Dataset contains {:d} total samples".format(len(self.rxns)))
        logging.info("{:d} max number of atoms ".format(self.max_n_all_atoms))
        self.degree_codec.fit(list(degrees))
        self.symbol_codec.fit(list(symbols))
        self.expl_val_codec.fit(list(explicit_valences))
        self.bond_type_codec.fit(list(bond_types))
        


    def __getitem__(self, idx):
        rxn = self.rxns[idx]
        react_smiles = rxn.reactants_smile
        
        rxn_feats = self.get_reaction_features(rxn)
        return rxn_feats

        
 
    def isfloat(self,value):
        try:
            float(value)
            return True
        except ValueError:
            return False 
    
        
    def get_reaction_features(self,rxn): 
        reactants=rxn.reactants 
        
        rxn_feats=defaultdict(lambda:defaultdict())        
        rxn_feats=self.get_molecule_features(mol_obj=None,smiles=rxn.reactants_smile)
        
        rxn_feats["yield_label"]=torch.tensor([rxn.yields])    
        #rxn_feats["domain_feats"]= rxn.domain
        rxn_feats['id'] = torch.tensor([rxn.Id])   
        rxn_feats["domain_feats"]=self.domain_feats[rxn.Id]
        return rxn_feats


    def get_molecule_features(self,mol_obj=None,smiles=None):
        output=defaultdict(lambda:defaultdict())
        if smiles!=None:
            mol=Chem.MolFromSmiles(smiles)
        else:  
            smiles=mol_obj.smile
            mol=Chem.MolFromSmiles(mol_obj.smile)
        atom_idx = torch.tensor([atom.GetIdx()-1 for atom in mol.GetAtoms()], dtype=torch.int64)

        n_atoms = mol.GetNumAtoms()
        sparse_idx = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                sparse_idx.append([atom_idx[i],atom_idx[j]])
                
        sparse_idx = torch.tensor(sparse_idx, dtype=torch.int64)
        atom_feats= self.get_atom_features(mol)
        bond_feats = self.get_bond_features(mol)
        atom_graph = torch.zeros((n_atoms,self.max_nbonds), dtype=torch.int64)
        bond_graph = torch.zeros((n_atoms, self.max_nbonds), dtype=torch.int64)
        n_bonds = torch.zeros((n_atoms,), dtype=torch.int32)

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()-1
            a2 = bond.GetEndAtom().GetIdx()-1
            idx = bond.GetIdx()
            if n_bonds[a1] == self.max_nbonds or n_bonds[a2] == self.max_nbonds:
                raise Exception(smiles)

            atom_graph[a1, n_bonds[a1]] = a2
            atom_graph[a2, n_bonds[a2]] = a1
            bond_graph[a1, n_bonds[a1]] = idx
            bond_graph[a2, n_bonds[a2]] = idx
            n_bonds[a1] += 1
            n_bonds[a2] += 1

        #binary_feats = self.get_binary_features(smiles)

        output["atom_feats"]=atom_feats
        output["bond_feats"]=bond_feats
        output["atom_graph"]=atom_graph
        output["bond_graph"]=bond_graph
        output["n_bonds"]=n_bonds
        output["n_atoms"]=n_atoms
        #output["binary_feats"]=binary_feats
        output["sparse_idx"]=sparse_idx
        #output["domain_feats"]=domain_features_final        
        return output  


    
    def get_atom_features(self,mol):
        atom_idx = torch.tensor([atom.GetIdx()-1 for atom in mol.GetAtoms()], dtype=torch.int64)

        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        expl_vals = [atom.GetExplicitValence() for atom in mol.GetAtoms()]

        t_symbol = self.to_one_hot(self.symbol_codec, symbols)
        t_degree = self.to_one_hot(self.degree_codec, degrees)
        t_expl_val = self.to_one_hot(self.expl_val_codec, expl_vals)
        t_aromatic = torch.tensor([atom.GetIsAromatic() for atom in mol.GetAtoms()]).float().unsqueeze(1)

        general_feats=torch.cat((t_symbol, t_degree, t_expl_val, t_aromatic), dim=1)[atom_idx]
        return general_feats

    def get_bond_features(self,mol):
        bond_types = [bond.GetBondType() for bond in mol.GetBonds()]
        t_bond_types = self.to_one_hot(self.bond_type_codec, bond_types)
        t_conjugated = torch.tensor([bond.GetIsConjugated() for bond in mol.GetBonds()]).float().unsqueeze(1)
        t_in_ring = torch.tensor([bond.IsInRing() for bond in mol.GetBonds()]).float().unsqueeze(1)
        return torch.cat((t_bond_types, t_conjugated, t_in_ring), dim=1)            
    
    def get_binary_features(self,smiles):

        comp = {}
        rxn_mol=Chem.MolFromSmiles(smiles)
        symbols = [atom.GetSymbol() for atom in rxn_mol.GetAtoms()]
        all_atoms=set(symbols)
        n_atoms=len(rxn_mol.GetAtoms())
        mapping=dict(zip( all_atoms, range(len(all_atoms))))      

        for i, atom in enumerate(rxn_mol.GetAtoms()):
            comp[i] = mapping[atom.GetSymbol()]
        n_comp = len(smiles.split('.'))
        rmol = Chem.MolFromSmiles(smiles)
        bond_map = {}

        for bond in rmol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bond_map[(a1, a2)] = bond
        binary_feats = torch.zeros((self.max_nbonds,self.max_natoms,self.max_natoms))
   
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if (i,j) in bond_map:
                    bond = bond_map[(i,j)]
               
                    binary_feats[1:1+6,i,j] = binary_feats[1:1+6,j,i] = self.bond_features(bond)
                elif (j,i) in bond_map:
                    bond = bond_map[(j,i)]
                    binary_feats[1:1+6,i,j] = binary_feats[1:1+6,j,i] = self.bond_features(bond)
                else:
                    binary_feats[0,i,j] = binary_feats[0,j,i] = 1.0
                #binary_feats[-2,i,j] = binary_feats[-2,j,i] = 1.0 if comp[i] != comp[j] else 0.0
                #binary_feats[-1,i,j] = binary_feats[-1,j,i] = 1.0 if comp[i] == comp[j] else 0.0
    
        return binary_feats
    
    @staticmethod
    def to_one_hot(codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_), dtype=torch.float)[value_idxs]

    @staticmethod
    def bond_features(bond):
        bt = bond.GetBondType()
        return torch.Tensor(
            [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()])
    
    


