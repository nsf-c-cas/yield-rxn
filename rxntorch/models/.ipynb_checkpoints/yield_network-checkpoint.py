import logging
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
from .wln import WLNet
from .attention import Attention
from .yield_scoring import YieldScoring
from sklearn.metrics import roc_auc_score,r2_score
from sklearn.metrics import mean_squared_error as s_mse
from sklearn.metrics import mean_absolute_error as s_mae

class YieldNet(nn.Module):
    def __init__(self, depth, dropout, afeats_size, bfeats_size, hidden_size, binary_size,dmfeats_size,max_nbonds,use_domain,abs_score):
        super(YieldNet, self).__init__()
        self.hidden_size = hidden_size
        self.wln = WLNet(depth, dropout, afeats_size, bfeats_size, hidden_size)
        
        self.attention = Attention(hidden_size, binary_size, max_nbonds)
        self.yield_scoring= YieldScoring(hidden_size, binary_size, dmfeats_size, use_domain,abs_score)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms, binary_feats, mask_neis, mask_atoms, sparse_idx,domain_feats):
        local_features = self.wln(fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms, mask_neis, mask_atoms)
        global_features = self.attention(local_features, binary_feats, sparse_idx)
        yield_scores = self.yield_scoring(local_features, global_features, binary_feats, sparse_idx, domain_feats)
        

        return yield_scores


class YieldTrainer(nn.Module):
    def __init__(self, rxn_net, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0, with_cuda=True,
                 cuda_devices=None, log_freq=10, grad_clip=None, pos_weight=1.0, lr_decay=0.9,
                 lr_steps=10000, max_nbonds=15):
        super(YieldTrainer, self).__init__()
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.model = rxn_net
        if cuda_condition and (torch.cuda.device_count() > 1):
            logging.info("Using {} GPUS".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.model.to(self.device)
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_steps = lr_steps
        self.grad_clip = grad_clip
        self.log_freq = log_freq
        self.pos_weight = pos_weight
        self.total_iters = 0
        self.max_nbonds= max_nbonds
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        self.lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True,factor=self.lr_decay,min_lr=1e-6)
        
    def train_epoch(self, epoch, data_loader):
        logging.info(80*"*"+'\n')
        logging.info("{:-^80}".format("Training"))
        self.model.train()
        r2 ,rmse,mae, w1,w2= self.iterate(epoch, data_loader, train=True,valid=False)
        return r2,rmse,mae,w1,w2

    def test_epoch(self, epoch, data_loader):
        logging.info("{:-^80}".format("Testing"))
        with torch.no_grad():
            self.model.eval()
            r2 ,rmse,mae= self.iterate(epoch, data_loader, train=False,valid=False)
        return r2,rmse,mae

    def valid_epoch(self, epoch, data_loader):
        logging.info("{:-^80}".format("Validating"))
        with torch.no_grad():
            self.model.eval()
        r2 ,rmse,mae= self.iterate(epoch, data_loader, train=False,valid=True)
        return r2,rmse,mae


    def iterate(self, epoch, data_loader, train=True,valid=False):
        avg_loss = test_loss = 0.0
        sum_gnorm = 0.0
        tmp_r2=0
        iters = len(data_loader)
        n_samples = len(data_loader.dataset)
        r2=0
        #correct_yields=np.array([[0]])
        #pred_yields=np.array([[0]])
        correct_yields=[]
        pred_yields=[]


        cum_r2=0
        for i, data in enumerate(data_loader):
            data = {key: value.to(self.device, non_blocking=True) for key, value in data.items()}

            # Create some masking logic for padding
            mask_neis = torch.unsqueeze(
                data['n_bonds'].unsqueeze(-1) > torch.arange(0, self.max_nbonds, dtype=torch.int32, device=self.device).view(1, 1, -1), -1)
            max_n_atoms = data['n_atoms'].max()
            mask_atoms = torch.unsqueeze(
                data['n_atoms'].unsqueeze(-1) > torch.arange(0, max_n_atoms, dtype=torch.int32, device=self.device).view(1, -1),
                -1)
            
            yield_scores = self.model.forward(data['atom_feats'], data['bond_feats'],data['atom_graph'], 
                                                data['bond_graph'], data['n_bonds'],data['n_atoms'], data['binary_feats'], 
                                                mask_neis, mask_atoms,data['sparse_idx'],data['domain_feats'])
            criteria=nn.MSELoss()

            
            loss= criteria(yield_scores, data['yield_label'])
            loss = torch.mean(loss)
            avg_loss += loss.item()
            test_loss += loss.item()

            aa=data['yield_label'].cpu().detach().numpy()
            bb=yield_scores.cpu().detach().numpy()
            #correct_yields=np.append(correct_yields,aa,0)
            #pred_yields=np.append(pred_yields,bb,0)
            correct_yields.append(aa)
            pred_yields.append(bb)
                
            if aa.shape !=bb.shape or len(correct_yields)!= len(pred_yields):
                raise ValueError("Found input variables with inconsistent numbers of elements")
                
            #tmp_r2 = r2_score(correct_yields,pred_yields)
            #cum_r2 += r2_score(aa,bb)
                
            #tmp_r2=0 if math.isnan(tmp_r2) else tmp_r2
            #cum_r2=0 if math.isnan(cum_r2) else cum_r2
            
            learning_rate=np.mean([ group['lr'] for group in self.optimizer.param_groups ])
            if train:
                self.total_iters += 1
                self.optimizer.zero_grad()
                loss.backward()
                
                param_norm = torch.sqrt(sum([torch.sum(param ** 2) for param in self.model.parameters()])).item()
                grad_norm = torch.sqrt(sum([torch.sum(param.grad ** 2) for param in self.model.parameters() if param.grad is not None])).item()
                sum_gnorm += grad_norm
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
            if (i+1) % self.log_freq == 0:
                
                    
                avg_loss = 0.0
                sum_gnorm = 0.0
        
        y_actual, y_pred = np.concatenate(np.array(correct_yields)).squeeze(1), np.concatenate(np.array(pred_yields)).squeeze(1)
        logging.info(f"y_actual, y_pred shape: {y_actual.shape},{y_pred.shape}")
        rmse= np.sqrt(s_mse(y_actual, y_pred ))
        mae= s_mae(y_actual, y_pred )
        final_r2 = r2_score(y_actual, y_pred )
        
        if valid:
            #logging.info("Epoch: {:2d}  valid Loss: {:f}  valid R2: {:6.2%}  LR:{:8.8f}  ".format(epoch,test_loss, tmp_r2,learning_rate))
            logging.info("Epoch: {:2d}  valid RMSE: {:f}  valid R2: {:6.2%}  valid MAE: {:6.2%}  LR:{:8.8f}  ".format(epoch,rmse,final_r2, mae, learning_rate))
            #return tmp_r2,avg_loss
            return final_r2,rmse,mae#,w1,w2
        
        if train:
            
            self.lr_scheduler.step(avg_loss)# for Su, with domain
            logging.info("Epoch: {:2d}  train RMSE: {:f}  train R2: {:6.2%}  train MAE: {:6.2%}  LR:{:8.8f} param norm: {:8.4f}  grad norm: {:8.4f} ".format(epoch,rmse, final_r2,mae, learning_rate,param_norm, sum_gnorm))
            
            learned_weights=self.model.module.yield_scoring.finalscore.weight.data
            w1,w2=learned_weights[0][0].item(),learned_weights[0][1].item()
            #w1,w2=0,0
            #return tmp_r2,avg_loss,w1,w2
            return final_r2,rmse,mae,w1,w2
        
        if not train and not valid:
            
            logging.info("Epoch: {:2d}  Test RMSE: {:f}  Test R2: {:6.2%}  test MAE: {:6.2%}  Test LR:{:8.8f}  ".format(epoch,rmse, final_r2,mae,learning_rate))
            #return tmp_r2, test_loss
            return final_r2,rmse,mae#,w1,w2

        

        
            
    def save(self, epoch, filename, path):
        output_path=os.path.join(path ,filename ,'yield.model')
        torch.save(self.model.module.cpu(), output_path)
        self.model.to(self.device)
        logging.info("Model saved to {}:".format(output_path))

