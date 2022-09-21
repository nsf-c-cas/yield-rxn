import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear


class Attention(nn.Module):
    def __init__(self, hidden_size, max_nbonds, max_natoms):
        super(Attention, self).__init__()
        self.fcapair = Linear(hidden_size, hidden_size, bias=False)
        self.fcattention = Linear(hidden_size, 1)
        cuda_condition= True
        self.device = torch.device("cuda" if cuda_condition else "cpu")
    def forward(self, local_feats, sparse_idx):
        local_pair = local_feats.unsqueeze(1) + local_feats.unsqueeze(2)
        
        local_features = self.fcapair(local_pair)
       
        d1, d2, d3, d4 = local_features.shape
        
        

        
        
        attention_features = F.relu( local_features)
        
        
        attention_score = torch.sigmoid(self.fcattention(attention_features))
        global_feats = torch.sum(local_feats.unsqueeze(1) * attention_score, dim=2)

        return global_feats 