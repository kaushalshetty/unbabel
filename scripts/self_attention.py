import torch
import torch.nn.functional as F
class SelfAttention(torch.nn.Module):
    def __init__(self,hid_dim):
        super(SelfAttention,self).__init__()
        self.k_dim = hid_dim
        self.q_projection = torch.nn.Linear(hid_dim,hid_dim)
        self.k_projection = torch.nn.Linear(hid_dim,hid_dim)
        self.v_projection = torch.nn.Linear(hid_dim,hid_dim)

    def forward(self,q,k,v):
        q_proj = self.q_projection(q)
        k_proj = self.k_projection(k)
        v_proj = self.v_projection(v)
        attention = F.softmax(q_proj@torch.t(k_proj),dim=1)
        att_wtd = (attention @ v_proj)
        return att_wtd/self.k_dim**0.5