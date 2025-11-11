# Copied and simplified from official https://github.com/microsoft/LoRA
# Try solving on your own without looking at the original sources

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.1,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Mark the weight as unmerged
        self.merged = False
        self.r = r

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()


    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass through the layer, incorporating LoRA adaptation if enabled.

        Args:
            x (torch.Tensor): The input tensor to the layer.

        Returns:
            torch.Tensor: The output tensor after applying the layer's operations.
        """

        # If LoRA adaptation is enabled and not merged:
        if self.r > 0 and not self.merged:

            # Apply the initial linear transformation using the base layer's weights
            transformed = F.linear(x, self.weight, bias=self.bias)

            # Apply dropout to the input.
            # Then multiply the dropout-applied input with the transposed lora_A matrix,
            # and then with the transposed lora_B matrix.
            # Then scale the LoRA adaptation and add it to the original transformation.
            ###### your code here: #####
            inp = self.lora_dropout(x)
            res = torch.matmul(inp, self.lora_A.transpose(0,1))
            res = torch.matmul(res, self.lora_B.transpose(0,1))
            transformed = self.scaling * res + transformed

            return transformed

        else:
            # Simply apply the base linear layer without LoRA modifications
            return F.linear(x, self.weight, bias=self.bias)


def compute_lora_linear(x: torch.Tensor, in_features: int, out_features: int, rank: int) -> torch.Tensor:
    torch.manual_seed(0)  # Ensure reproducibility
    model = LoRALinear(in_features, out_features, rank)

    # Simulate training by initializing lora_A and lora_B with non-zero values
    # Normally, these values would be learned during the training process
    with torch.no_grad():
        model.lora_A.copy_(torch.randn_like(model.lora_A))
        model.lora_B.copy_(torch.randn_like(model.lora_B))

    return model.forward(x)