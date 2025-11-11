import torch
import torch.autograd as autograd

class CustomReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for the ReLU activation function.
        """
        # Save the input tensor 'x' for use in the backward pass
        ctx.save_for_backward(x)

        ###### your code here: #####
        x = torch.clamp(x, min=0)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the ReLU activation function.
        """
        # Retrieve the saved input tensor
        input, = ctx.saved_tensors

        # Cloning is necessary because directly modifying grad_output could affect other
        # parts of the computational graph that are using this tensor.
        grad_input = grad_output.clone()

        ###### your code here: #####
        '''
        Hint: Usually computation looks something like this:
        input, = ctx.saved_tensors
        # Compute the derivative of the function with respect to input
        derivative = f_prime(input)
        # Multiply by grad_output (chain rule)
        grad_input = grad_output * derivative
        '''
        # grad_input = grad_output * 1 if (input > 0) else 0
        mask = input > 0

        grad_input = grad_output * mask.to(dtype=grad_output.dtype)

        return grad_input


def compute_custom_relu(input_tensor: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Apply the CustomReLUFunction to the input tensor and compute the gradient.

    Args:
        input_tensor (torch.Tensor): The input tensor for which forward and backward passes are computed.

    Returns:
        tuple: A tuple containing the output tensor after applying ReLU activation and
               the gradient tensor after the backward pass.
    """
    # Apply the custom ReLU function
    relu = CustomReLUFunction.apply
    # Make a copy of the input tensor and set requires_grad to True for automatic differentiation
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    # Compute the ReLU output
    output = relu(input_tensor)
    # Initiate the backward pass with a gradient tensor of ones
    output.backward(torch.ones_like(input_tensor))
    # Return both the ReLU output and the gradient of the input tensor
    return output, input_tensor.grad