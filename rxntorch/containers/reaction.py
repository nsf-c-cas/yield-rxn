from __future__ import print_function

#from .molecule import Mol
import rdkit.Chem.AllChem as AllChem

class Rxn(object):
    """
    """
    def __init__(self,Id,reactants,solvent,base,product,r_yield):
        self.yields = float(r_yield)/100
        self.reactants = '.'.join([reactants,solvent,base])  
        self.product = product
        self.Id= int(Id)
        #self.domain = domain_features
        self.reactants_smile = self.reactants 

    
    def get_yield(self):
        return self.yields
    
    @property
    def smile(self):
        return self.reactants_smile
        #return '>'.join([self.reactants_smile, self.products_smile])
    
    

