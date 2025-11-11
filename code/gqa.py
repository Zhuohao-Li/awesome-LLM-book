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
        assert self.num_heads % self.n_kv_heads == 0
        n_rep = self.num_heads // self.n_kv_heads
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=2)
        # (bsz, seqlen, n_kv_heads, head_dim) -> (bsz, seqlen, num_heads, head_dim)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        # x: (bsz, seqlen, dim)

        q = self.wq(x) # (bsz, seqlen, num_heads * head_dim)
        k = self.wk(x) # (bsz, seqlen, n_kv_heads * head_dim)
        v = self.wv(x) # (bsz, seqlen, n_kv_heads * head_dim)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim) # (bsz, seqlen, num_heads, head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim) # (bsz, seqlen, n_kv_heads, head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim) # (bsz, seqlen, n_kv_heads, head_dim)

        k = self.repeat_kv(k) # (bsz, seqlen, num_heads, head_dim)
        v = self.repeat_kv(v) # (bsz, seqlen, num_heads, head_dim)

        q = q.transpose(1, 2) # (bsz, num_heads, seqlen, head_dim)
        k = k.transpose(1, 2) # (bsz, num_heads, seqlen, head_dim)
        v = v.transpose(1, 2) # (bsz, num_heads, seqlen, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (bsz, num_heads, seqlen, seqlen)
        attn = F.softmax(scores, dim=-1) # (bsz, num_heads, seqlen, seqlen)

        out = torch.matmul(attn, v) # (bsz, num_heads, seqlen, head_dim)
        out = out.transpose(1, 2) # (bsz, seqlen, num_heads, head_dim)
        out = out.reshape(bsz, seqlen, -1) # (bsz, seqlen, dim)
        out = self.wo(out) # (bsz, seqlen, dim)
        return out