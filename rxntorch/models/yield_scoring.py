import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear


class YieldScoring(nn.Module):
    def __init__(self, hidden_size, dmfeats_size, use_domain,abs_score):
        super(YieldScoring, self).__init__()
        self.use_domain= use_domain

        self.fclocal = Linear(hidden_size, hidden_size, bias=False)
        self.fcglobal = Linear(hidden_size, hidden_size)
        self.fcscore = Linear(hidden_size, 1)
        
        self.domain= Linear(dmfeats_size, 1)
        
        self.finalscore = Linear(2,1)
        
        
    def forward(self, local_features, global_features, sparse_idx,domain_feats):
      
        
        global_feats= self.fcglobal(global_features)
        local_feats= self.fclocal(local_features)
        d1, d2, d3 = global_feats.shape
        
        #binary_feats = binary.reshape(l1, l2, l3*l4)
        

        features= F.relu(global_feats +local_feats )# + self.fcbinary(binary_feats))
        
        l1, l2 = domain_feats.shape        

        if self.use_domain=='no_rdkit' or self.use_domain=='rdkit':
            
            score_d= self.domain( domain_feats)
            score_g = self.fcscore(features)

            score_g_avg=torch.mean(torch.abs(score_g),dim=1)
            score_d_avg=torch.abs(score_d)
            #final_score=(score_d_avg+score_g_avg)/2
            final_score=self.finalscore(torch.cat([score_d_avg,score_g_avg],dim=1))
            return final_score
        else:
            score_g = self.fcscore(features)
            final_score=torch.mean(torch.abs(score_g),dim=1)
            return final_score
            
        



 