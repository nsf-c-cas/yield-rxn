from __future__ import print_function
from __future__ import division

import os
import logging
import rdkit.Chem as Chem
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import json
from collections import defaultdict
import math
from .reaction import Rxn
import re
import glob


class RxnDataset(Dataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_name,path="data/", vocab=None):
        super(RxnDataset, self).__init__()
        self.file_name = file_name
        self.path = path
        self.vocab = vocab

    def __len__(self):
        return len(self.rxns)

    def __setitem__(self, idx, value):
        self.rxns[idx] = value

    @property
    def rxn_smiles(self):
        return [rxn.smile for rxn in self.rxns]

    @rxn_smiles.setter
    def rxn_smiles(self, rxn_smiles):
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]

    @classmethod
    def from_list(cls, rxns):
        new_dataset = cls('')
        new_dataset.rxns = rxns
        return new_dataset

    def save_to_file(self, file_name=None, path=None):
        if file_name == None:
            file_name = self.file_name
        if path == None:
            path = self.path
        with open(os.path.join(path, file_name), "w") as f:
            for rxn in self.rxn_smiles:
                f.write(rxn+"\n")

    def canonicalize(self):
        for rxn in self.rxns:
            rxn.canonicalize()

    def remove_rxn_mappings(self):
        for rxn in self.rxns:
            rxn.remove_rxn_mapping()

    def remove_max_reactants(self, max_reactants):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.reactants) <= max_reactants]
        self.rxns = keep_rxns

    def remove_max_products(self, max_products):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.products) <= max_products]
        self.rxns = keep_rxns


        
class RxnGraphDataset(RxnDataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.
    """    
    def __init__(self, file_name, path, use_domain):
        super(RxnGraphDataset, self).__init__(file_name, path)
        self.file_name = file_name
        if use_domain=='no_rdkit':
            self.domain_file = os.path.join(path,file_name.split('.')[0]+'_raw_nordkit.txt')
        else: #if it's rdkit or False. If 
            self.domain_file = os.path.join(path,file_name.split('.')[0]+'.txt')
            
            
        self.data_name= file_name.split('/')[0]
        #logging.info(file_name, self.domain_file,mol_path,path)
        self.path = path
        self.rxns = []
        self.degree_codec = LabelEncoder()
        self.symbol_codec = LabelEncoder()
        self.expl_val_codec = LabelEncoder()
        self.bond_type_codec = LabelEncoder()
        self.mol_path =os.path.join(self.path, self.data_name, self.data_name+'_reaction_mols')        
        
        self.max_nbonds = 15
        self.max_natoms = 15
        self.max_n_all_atoms=0
        max_min_dic={'su': [-184,634],
                      'az': [],
                     'dy':[-116,514]}
            
        
        self._init_dataset()
        
    def _init_dataset(self):
        logging.info("Loading Dataset {dataset} in {datapath}".format(
            dataset=self.file_name, datapath=self.path))
        symbols = set()
        degrees = set()
        explicit_valences = set()
        bond_types = set()
        i=0
        
        #build domain dictionary
        self.domain_feats=defaultdict(list)      
        lines=open(self.domain_file,'r').readlines()                    
        
        for line in lines:
            key,vals=line.strip('\n').split('\t')
            self.domain_feats[str(key)]= [float(i) for i in vals.split(',')]
        
        for data_type in ['su','az','dy']:
            file_name = data_type+'_reactions_data.json'
            
            with open(os.path.join(self.path, data_type,file_name)) as datafile:
            #file_name = self.data_name+'_reactions_data.json'
            #data_type = self.data_name
            #with open(os.path.join(self.path, self.data_name,file_name)) as datafile:
            

                data = json.load(datafile)
                lines= data
                solvent_key='solvent' if 'solvent' in lines[0].keys() else 'Solvent'
                base_key= 'Base' if 'Base' in lines[0].keys() else 'base'
                
                
                for line in lines:
                        
                    if 'az' in file_name:
                        name= reaction_num = line["reaction_Num"]
                        r_yield=line['yield']['yield']

                    elif 'su' in file_name or 'dy' in file_name:
                        name= reaction_num = line["Id"]
                        r_yield=line['yield']
                    
                    reactants=line['reactants']
                    base = line.get(base_key,{}).get('smiles','')
                    solvent=  line.get(solvent_key,[''])[0]
                    if 'dy' in file_name:
                        solvent='CS(=O)C'
                    rxn = Rxn(name,reactants,solvent,base,r_yield)
                    reactants=rxn.reactants

                    n_atoms=Chem.MolFromSmiles(rxn.reactants_smile).GetNumAtoms()
                    self.max_n_all_atoms = max(self.max_n_all_atoms, n_atoms)

                    if data_type in self.file_name:
                        i+=1
                        self.rxns.append(rxn)
                    mol = Chem.MolFromSmiles(rxn.reactants_smile)

                    for atom in mol.GetAtoms():   
                        symbols.add(atom.GetSymbol())
                        degrees.add(atom.GetDegree())
                        explicit_valences.add(atom.GetExplicitValence())
                    for bond in mol.GetBonds():
                        bond_types.add(bond.GetBondType())

            symbols.add("unknown")
            logging.info("Dataset contains {:d} total samples".format(len(self.rxns)))
            logging.info("{:d} number of appends ".format(i))
            logging.info("{:d} max number of atoms ".format(self.max_n_all_atoms))
        self.degree_codec.fit(list(degrees))
        self.symbol_codec.fit(list(symbols))
        self.expl_val_codec.fit(list(explicit_valences))
        self.bond_type_codec.fit(list(bond_types))
        


    def __getitem__(self, idx):
        rxn = self.rxns[idx]
        react_smiles = rxn.reactants_smile
        
        rxn_feats,mod_idx_feats=self.get_reaction_features(rxn)
        return rxn_feats

    max_num_reactants=4
        
 
    def isfloat(self,value):
        try:
            float(value)
            return True
        except ValueError:
            return False 
    
        
    def get_reaction_features(self,rxn): #rxn is a Rxn object
        reactants=rxn.reactants # is a mol_obj
        #rxn_feats=self.get_molecule_features(reactants[0])#initialized with the first reactant
        rxn_feats=defaultdict(lambda:defaultdict())
        #domain_feats=self.normalize_mol_attributes(reactants[0].get_attributes())

        domain_feats=torch.tensor(list(self.domain_feats[str(rxn.Id)])).unsqueeze(0)
        mol_idx_feat=defaultdict(lambda:defaultdict())
        
        rxn_feats=self.get_molecule_features(mol_obj=None,smiles=rxn.reactants_smile)
        rxn_feats["yield_label"]=torch.tensor([rxn.yields])
        
            
        rxn_feats["domain_feats"]=domain_feats
        
        return rxn_feats,mol_idx_feat


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

        binary_feats = self.get_binary_features(smiles)

        output["atom_feats"]=atom_feats
        output["bond_feats"]=bond_feats
        output["atom_graph"]=atom_graph
        output["bond_graph"]=bond_graph
        output["n_bonds"]=n_bonds
        output["n_atoms"]=n_atoms
        output["binary_feats"]=binary_feats
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
        n_atoms=len(all_atoms)

        mapping=dict(zip( all_atoms, range(len(all_atoms))))

        for i, s in enumerate(smiles.split('.')):
            mol = Chem.MolFromSmiles(s)
            for atom in mol.GetAtoms():
                comp[mapping[atom.GetSymbol()]] = i

        n_comp = len(smiles.split('.'))
        rmol = Chem.MolFromSmiles(smiles)
        bond_map = {}
        for bond in rmol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()- 1
            a2 = bond.GetEndAtom().GetIdx() - 1
            bond_map[(a1, a2)] = bond

        binary_feats = torch.zeros((self.max_nbonds,self.max_natoms, self.max_natoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if i == j:
                    continue
                if (i,j) in bond_map:
                    bond = bond_map[(i,j)]
                    binary_feats[1:1+6,i,j] = binary_feats[1:1+6,j,i] = self.bond_features(bond)
                elif (j,i) in bond_map:
                    bond = bond_map[(j,i)]
                    binary_feats[1:1+6,i,j] = binary_feats[1:1+6,j,i] = self.bond_features(bond)
                else:
                    binary_feats[0,i,j] = binary_feats[0,j,i] = 1.0
                binary_feats[-4,i,j] = binary_feats[-4,j,i] = 1.0 if comp[i] != comp[j] else 0.0
                binary_feats[-3,i,j] = binary_feats[-3,j,i] = 1.0 if comp[i] == comp[j] else 0.0
                binary_feats[-2,i,j] = binary_feats[-2,j,i] = 1.0 if n_comp == 1 else 0.0
                binary_feats[-1,i,j] = binary_feats[-1,j,i] = 1.0 if n_comp > 1 else 0.0
        return binary_feats
    
    @staticmethod
    def to_one_hot(codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_), dtype=torch.float)[value_idxs]

    #don't need this. In predicting molecule structure, the lables are bonds. Here we use yield. 
    @staticmethod
    def get_bond_labels(edits, n_atoms, sparse_idx):
        bo_index_map = {0.0: 0,
                           1: 1,
                           2: 2,
                           3: 3,
                           1.5: 4}
        edits = edits.split(";")
        bond_labels = torch.zeros((n_atoms, n_atoms, len(bo_index_map)))
        for edit in edits:
            atom1, atom2, bond_order = edit.split("-")
            bo_index = bo_index_map[float(bond_order)]
            bond_labels[int(atom1)-1, int(atom2)-1, bo_index] = bond_labels[int(atom2)-1, int(atom1)-1, bo_index] = 1
        bond_labels = bond_labels[sparse_idx[:,0],sparse_idx[:,1]]
        return bond_labels

    @staticmethod
    def bond_features(bond):
        bt = bond.GetBondType()
        return torch.Tensor(
            [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()])
    
    
    def get_atom_mapping_doyle(self,mol_dir):
    
        smiles_mapping=defaultdict(dict) 

        for mol_fn in glob.glob(mol_dir):
            atoms , labels, atom_mapping =[], [], []
            m = Chem.MolFromMolFile(mol_fn)
            smiles=Chem.MolToSmiles(m)
            mol_lines=open(mol_fn,'r').readlines()

            for i,line in enumerate(mol_lines[4:]):
                l=re.sub(' +', ' ', line.strip('\n'))
                l2=l.split(' ')
                if len(l2)==17:
                    atom=l2[4]
                    if atom !='' and '*' not in atom and 'H' not in atom:
                        atoms.append(atom)
                if "atom_labels" in line:
                    labels=[i.strip('*') for i in mol_lines[i+1+4].strip('\n').split(' ') if ((i!='') and ('H' not in i))]
            if len(atoms)==len(labels):
                for i in range(len(atoms)):
                    atom_mapping.append((atoms[i],labels[i]))            
            else:
                print('Atoms and lables don\'t match')
                atom_mapping=[]
            if smiles not in smiles_mapping:
                smiles_mapping[mol_fn.split('/')[-1].split('.')[0]]=atom_mapping
            else:
                print("key exsists")
        return smiles_mapping


    def get_atom_domain_features(self,mol_obj):
        p_charge_dict=defaultdict(float)
        nmr_dict=defaultdict(float)
        
        for atom in mol_obj.atoms: # first get all partial charges for different atom labels
            if 'partial_charge' in atom:
                p_charge_dict[atom['name']]=atom['partial_charge']
            if 'nmr_shift' in atom:
                nmr_dict[atom['name']]=self.normalize(atom['nmr_shift'],self.shifts)
                
    
        partial_charges = []
        nmr_shifts = []
        mol = Chem.MolFromMolFile(self.mol_path+'/'+mol_obj.name+'.mol')
        
        for atom in mol.GetAtoms():
            atom_label = self.smiles_mapping[mol_obj.name][atom.GetIdx()][1]
            partial_charges.append(p_charge_dict[atom_label])
            nmr_shifts.append(nmr_dict[atom_label])

        nmr_shifts = torch.tensor(nmr_shifts).float()
        partial_charges  = torch.tensor(partial_charges).float()
        
        domain_feats=torch.cat((nmr_shifts,partial_charges),dim=0).float().unsqueeze(0)
        return domain_feats

