import torch

def solution(data: torch.Tensor, epsilon) -> torch.Tensor:
    result = None
    ###### your code here: ##### 
    result = (data - torch.mean(data, dim=0, keepdim=True)) / (torch.sqrt(torch.var(data, dim=0, keepdim=True) + epsilon))
    return result
