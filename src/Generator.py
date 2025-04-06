import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def renormalize_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Renormalizes the probabilities to sum to 1.
    """
    return probs / probs.sum(dim=-1, keepdim=True)

def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """
    Applies temperature scaling to the logits.
    
    Args:
        logits: Raw model logits [batch_size, vocab_size]
        temperature: Temperature values [batch_size, 1]
        
    Returns:
        Temperature-scaled logits
    """
    return logits / temperature

def apply_nucleus(probs: torch.Tensor, nucleus: torch.Tensor) -> torch.Tensor:
    """
    Applies nucleus (top-p) sampling to the probability distribution.
    
    Args:
        probs: Probability distribution [batch_size, vocab_size]
        nucleus: Probability threshold values [batch_size, 1]
        
    Returns:
        Filtered probability distribution with same shape
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices to remove (where cumulative probability exceeds nucleus threshold)
    sorted_indices_to_remove = cumulative_probs > nucleus
    
    # Shift indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False  # Always keep the most probable token
    
    # Zero out the probabilities to remove
    sorted_probs[sorted_indices_to_remove] = 0.0

    # Scatter the sorted filtered values back to their original positions
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_indices, sorted_probs)
    
    return renormalize_probs(filtered_probs)

def get_probs(logits: torch.Tensor, temperature: torch.Tensor, nucleus: torch.Tensor) -> torch.Tensor:
    """
    Processes model logits to get the filtered probability distribution.
    
    Args:
        logits: Raw model logits [batch_size, vocab_size]
        temperature: Temperature values [batch_size]
        nucleus: Nucleus sampling threshold values [batch_size]
        
    Returns:
        Filtered probability distribution
    """
    batch_size, vocab_size = logits.shape
    assert nucleus.shape[0] == batch_size, "nucleus must have size [batch_size]"
    assert temperature.shape[0] == batch_size, "temperature must have size [batch_size]"

    # Reshape parameters to [batch_size, 1] for broadcasting
    temperature = temperature.view(-1, 1)
    nucleus = nucleus.view(-1, 1)
    
    # Apply temperature scaling
    scaled_logits = apply_temperature(logits, temperature)
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Apply nucleus sampling
    probs = apply_nucleus(probs, nucleus)
    
    return probs

def select_next_token(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Selects the next token based on the probabilities and p-value threshold.
    
    Args:
        probs: Filtered probability distribution [batch_size, vocab_size]
        p: Random threshold for token selection
        
    Returns:
        Selected token IDs [batch_size, 1]
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cum_probs = torch.cumsum(sorted_probs, dim=-1) - p
    
    # Get the index of the first positive value
    first_positive_index = (cum_probs > 0).int().argmax(dim=-1)
    
    # Get the token ID at the first positive value
    batch_indices = torch.arange(probs.size(0), device=probs.device)
    next_tokens = sorted_indices[batch_indices, first_positive_index]
    
    return next_tokens.unsqueeze(1)

class Generator(pl.LightningModule):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.save_hyperparameters()

    @torch.no_grad()
    def generate_tokens(self, num_samples: int, max_length: int, p: torch.Tensor, 
                         temperature: torch.Tensor, nucleus: torch.Tensor,
                         prompt: str = None):
        """
        Generates token sequences in parallel with KV caching for speed improvement.
        
        Args:
            num_samples: Number of sequences to generate
            max_length: Maximum sequence length
            p: Random threshold values for token selection
            temperature: Temperature values for each sample
            nucleus: Nucleus threshold values for each sample
            prompt: Optional text prompt to start generation
            
        Returns:
            Generated token sequences
        """
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
        
        # Initialize KV cache as None
        past_key_values = None

        for i in range(remaining_tokens):
            # Pass the past_key_values and only the last token if we have a cache
            if past_key_values is not None:
                outputs = self.model(input_ids=generated_tokens[:, -1:], past_key_values=past_key_values, use_cache=True)
            else:
                outputs = self.model(input_ids=generated_tokens, use_cache=True)
            
            # Update KV cache
            past_key_values = outputs.past_key_values
            
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token

            probs = get_probs(next_token_logits, temperature, nucleus)
            next_tokens = select_next_token(probs, p[i])

            # Append the new token to the generated sequence
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

        return generated_tokens

    @torch.no_grad()
    def generate_text(self, num_samples: int, max_length: int, p: torch.Tensor, 
                      temperature: torch.Tensor = None, nucleus: torch.Tensor = None,
                      prompt: str = None, batch_size: int = None):
        """
        Generates text samples in parallel with optional batching.
        
        Args:
            num_samples: Number of sequences to generate
            max_length: Maximum sequence length
            p: Random threshold values for token selection
            temperature: Temperature values for each sample
            nucleus: Nucleus threshold values for each sample
            prompt: Optional text prompt to start generation
            batch_size: Process in smaller batches if provided
            
        Returns:
            List of generated text sequences
        """
        if temperature is None:
            temperature = torch.ones(num_samples, device=self.device)
        if nucleus is None:
            nucleus = torch.ones(num_samples, device=self.device)
            
        # Use the full batch if batch_size is None or greater than num_samples
        if batch_size is None or batch_size >= num_samples:
            # Process all samples at once
            print(f"Prompt generate_text no batch: {prompt}")

            generated_tokens = self.generate_tokens(
                num_samples, max_length, p, temperature, nucleus, prompt
            )
            decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            return decoded_texts
        
        # Process in batches
        all_decoded_texts = []
        
        for i in tqdm(range(0, num_samples, batch_size)):
            end_idx = min(i + batch_size, num_samples)
            current_batch_size = end_idx - i
                        
            # Get the batch-specific tensors
            batch_temperature = temperature[i:end_idx]
            batch_nucleus = nucleus[i:end_idx]
            
            # Generate tokens for this batch
            batch_tokens = self.generate_tokens(
                current_batch_size, max_length, p, batch_temperature, batch_nucleus, prompt
            )
            
            # Decode and store
            batch_texts = self.tokenizer.batch_decode(batch_tokens, skip_special_tokens=False)
            all_decoded_texts.extend(batch_texts)
        
        return all_decoded_texts