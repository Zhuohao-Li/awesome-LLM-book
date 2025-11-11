import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, dim, num_heads, n_kv_heads):
        super().__init__()
        self.dim = dim # embedding dimension
        self.num_heads = num_heads # number of query heads
        self.n_kv_heads = n_kv_heads # number of key/value heads
        self.head_dim = dim // num_heads # dimension of each head

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def repeat_kv(self, x):
        n_rep = self.num_heads // self.n_kv_heads
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=2)
        # (bsz, seqlen, n_kv_heads, head_dim) -> (bsz, seqlen, num_heads, head_dim)

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        q = q.transpose(1, 2)
        k = k.permute(0, 2, 1, 3)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, seqlen, -1)
        out = self.wo(out)
        return out