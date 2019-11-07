import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from feat.utils import euclidean_metric
from scipy.io import loadmat

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, args, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(nn.Module):

    def __init__(self, args, dropout=0.2):
        super().__init__()
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
            z_dim = 64
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
            z_dim = 640
        elif args.model_type == 'AmdimNet':
            from feat.networks.amdimnet import AmdimNet
            self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
            z_dim = args.rkhs
        else:
            raise ValueError('')

        self.slf_attn = MultiHeadAttention(args, args.head, z_dim, z_dim, z_dim, dropout=dropout)    
        self.z_dim = z_dim
        self.args = args

    def forward(self, support, query, mode = 'test'):
        # feature extraction
        support = self.encoder(support) # 5 x 2048 x  1 x 1
        # get mean of the support
        proto = support.reshape(self.args.shot, -1, support.shape[-1]).mean(dim=0) # N x d
        num_proto = proto.shape[0]
        # for query set
        query = self.encoder(query)
        
        # adapt the support set instances
        proto = proto.unsqueeze(0)  # 1 x N x d        
        # refine by Transformer
        proto = self.slf_attn(proto, proto, proto)
        proto = proto.squeeze(0)
        
        # compute distance for all batches
        logitis = euclidean_metric(query, proto) / self.args.temperature
        
        # transform for all instances in the task
        if mode == 'train':
            aux_task = torch.cat([support.reshape(self.args.shot, -1, support.shape[-1]), 
                                  query.reshape(self.args.query, -1, support.shape[-1])], 0) # (K+Kq) x N x d
            aux_task = aux_task.permute([1,0,2])
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # N x (K+Kq) x d
            # compute class mean
            aux_center = torch.mean(aux_emb, 1) # N x d
            logitis2 = euclidean_metric(aux_task.permute([1,0,2]).view(-1, self.z_dim), aux_center) / self.args.temperature2
            return logitis, logitis2
        else:
            return logitis