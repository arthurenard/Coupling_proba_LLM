import torch
def generate_p_values(max_length: int) -> torch.Tensor:
    """
    Generates random p-values for token selection.
    
    Args:
        max_length: Maximum sequence length to generate
        
    Returns:
        Random p-values [max_length - 1]
    """
    return torch.rand(max_length - 1)

def generate_temperature(num_samples: int) -> torch.Tensor:
    """
    Generates linearly spaced temperature values.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Temperature values [num_samples]
    """
    eps = 1e-6
    return torch.linspace(eps, 1.0, num_samples)

def generate_nucleus(num_samples: int) -> torch.Tensor:
    """
    Generates linearly spaced nucleus threshold values.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Nucleus threshold values [num_samples]
    """
    eps = 1e-6
    return torch.linspace(eps, 1.0, num_samples)
