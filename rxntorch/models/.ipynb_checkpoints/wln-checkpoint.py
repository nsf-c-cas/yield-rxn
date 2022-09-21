import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear
        
        
class WLNet(nn.Module):
    def __init__(self, depth, dropout, afeats_size, bfeats_size, hidden_size):
        super(WLNet, self).__init__()
        self.depth = depth
        #print("afeats_size;", afeats_size)
        self.fc1 = Linear(afeats_size, hidden_size, bias=False)
        self.graph_conv_nei = Linear(hidden_size + bfeats_size, hidden_size)
        self.graph_conv_atom = Linear(hidden_size * 2, hidden_size)
        self.fc2atom_nei = Linear(hidden_size, hidden_size, bias=False)
        self.fc2bond_nei = Linear(bfeats_size, hidden_size, bias=False)
        self.fc2 = Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, atom_feats_1, bond_feats, atom_graph, bond_graph, num_nbs, n_atoms, mask_neis, mask_atoms):
        #print("atom_feats_1.shape: ",atom_feats_1.shape)
        atom_feats = F.relu(self.fc1(atom_feats_1))
        bondnei_feats = torch.stack([bond_feats[i,bond_graph[i],:] for i in range(atom_feats.shape[0])])
        
        for i in range(self.depth):
            atomnei_feats = torch.stack([atom_feats[i,atom_graph[i],:] for i in range(atom_feats.shape[0])])
            if (i + 1) == self.depth:
                atomnei_feats = self.dropout(self.fc2atom_nei(atomnei_feats))
                bondnei_feats = self.dropout(self.fc2bond_nei(bondnei_feats))
                
                
                nei_feats = atomnei_feats * bondnei_feats
                nei_feats = torch.where(mask_neis, nei_feats, torch.zeros_like(nei_feats)).sum(-2)
                
                self_feats = self.fc2(atom_feats)
                local_feats = self_feats * nei_feats
                local_feats = torch.where(mask_atoms, local_feats, torch.zeros_like(local_feats))
                
            else:
                nei_feats = F.relu(self.graph_conv_nei(torch.cat([atomnei_feats, bondnei_feats], dim=-1)))
                nei_feats = torch.where(mask_neis, nei_feats, torch.zeros_like(nei_feats)).sum(-2)
                update_feats = torch.cat([atom_feats, nei_feats], dim=-1)
                atom_feats = F.relu(self.graph_conv_atom(update_feats))
        return local_feats
