
import torch
import math
import torch.nn.functional as F

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    result = None
    # ###### your code here: ##### 
    # v = query * torch.transpose(key)
    # v = v / torch.sqrt(torch.shape(key))
    # result = F.softmax(v)
    # result = result * value
    # return result


    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    # scores = scores - scores.max(dim=-1, keepdim=True).values
    attn = F.softmax(scores, dim=-1)
    result = torch.matmul(attn, value)
    return result