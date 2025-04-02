import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_probs(logits: torch.Tensor, temperature: torch.Tensor, nucleus: torch.Tensor) -> torch.Tensor:
    """
    Applies temperature scaling to the logits and returns the probabilities.
    Temperature is a tensor of shape [batch_size].
    """
    batch_size = logits.shape[0]
    assert nucleus.shape[0] == batch_size, "nucleus must have size [batch_size]"
    assert temperature.shape[0] == batch_size, "temperature must have size [batch_size]"

    # Reshape temperature to [batch_size, 1] for broadcasting
    temperature = temperature.view(-1, 1)
    nucleus = nucleus.view(-1, 1)
    
    # Apply temperature scaling
    next_token_logits = logits / temperature

    # Calculate probabilities
    probs = F.softmax(next_token_logits, dim=-1)
    
    # Apply nucleus to the sorted probabilities
    probs = probs * nucleus

    return probs

def select_next_token(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Selects the next token based on the probabilities and the p values.
    """
    # Sort the probabilities and get the indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Calculate cumulative probabilities
    cum_probs = torch.cumsum(sorted_probs, dim=1) - p

    # Get the index of the first positive value
    first_positive_index = (cum_probs > 0).int().argmax(dim=1)

    # Get the token ID at the first True value in the mask
    next_tokens = sorted_indices[torch.arange(sorted_indices.size(0)), first_positive_index]
    
    return next_tokens.unsqueeze(1)

def generate_p_values(max_length: int) -> torch.Tensor:
    """
    Returns the p ([max_length - 1]) values for the given max_length.
    """
    
    p_values = torch.rand(max_length - 1)

    return p_values

def generate_temperature(num_samples: int) -> torch.Tensor:
    """
    Returns the temperature ([num_samples]) values.
    """

    return torch.linspace(0.0, 1.2, num_samples)

def generate_nucleus(num_samples: int) -> torch.Tensor:
    """
    Returns the nucleus ([num_samples]) values.
    """

    return torch.linspace(0.0, 1.0, num_samples)

def generate_ones(num_samples: int) -> torch.Tensor:
    """
    Returns the ones ([num_samples]) values.
    """

    return torch.ones(num_samples)

class Generator(pl.LightningModule):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.save_hyperparameters()

    @torch.no_grad()
    def generate_tokens(self, num_samples: int, max_length: int, p: torch.Tensor, temperature: torch.Tensor, nucleus: torch.Tensor, prompt: str = None):
        """Generates text samples in parallel."""
        self.model.eval()  # Set model to evaluation mode

        # Handle input tokens based on prompt or default token
        if prompt:
            # Tokenize the prompt
            input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            # Repeat for each sample in the batch
            input_ids = input_tokens.repeat(num_samples, 1)
        else:
            # Get the start token ID (<|endoftext|>)
            start_token_id = self.tokenizer.eos_token_id
            if start_token_id is None:
                raise ValueError("Model tokenizer does not have an EOS token defined.")
            input_ids = torch.full((num_samples, 1), start_token_id, dtype=torch.long, device=self.device)
        
        generated_tokens = input_ids

        # Adjust remaining tokens to generate based on prompt length
        remaining_tokens = max_length - input_ids.size(1)
        if remaining_tokens <= 0:
            return input_ids[:, :max_length]  # Truncate to max_length if prompt is too long
        
        # Only use p values for the remaining tokens
        p = p[input_ids.size(1) - 1:]

        for i in tqdm(range(remaining_tokens)):
            outputs = self.model(input_ids=generated_tokens)
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token

            probs = get_probs(next_token_logits, temperature, nucleus)
            next_tokens = select_next_token(probs, p[i])

            # Append the new token to the generated sequence
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

        return generated_tokens

    @torch.no_grad()
    def generate_text(self, num_samples: int, max_length: int, p: torch.Tensor, temperature: torch.Tensor = None, nucleus: torch.Tensor = None, prompt: str = None):
        """Generates text samples in parallel."""
        
        # Tokens generation
        generated_tokens = self.generate_tokens(num_samples, max_length, p, temperature, nucleus, prompt)

        # Decode the generated sequences
        decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        return decoded_texts