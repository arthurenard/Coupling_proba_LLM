import os
import argparse
import torch
from pathlib import Path
import json

from src.Generator import Generator
from src.generate_values import generate_p_values, generate_temperature, generate_nucleus

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate text using multiple models and sampling methods.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of text samples to generate per model/method.")
    parser.add_argument("--prompts", nargs='+', required=True, help="List of prompts to use for generation.")
    parser.add_argument("--devices", type=str, default='1', help="Devices to use ('auto', comma-separated integers like '0' or '0,1'). Defaults to using the primary GPU if available.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base directory for outputs.")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum length of generated sequences.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for large models. If not provided, an appropriate batch size will be selected based on the model.")
    args = parser.parse_args()

    # Models to use, with batch size recommendations for large models
    models = [
        {"name": "gpt2", "dir_name": "gpt2", "batch_size": 20},  # Small model, use full batch
        {"name": "gpt2-medium", "dir_name": "gpt2-medium", "batch_size": 20},
        {"name": "gpt2-large", "dir_name": "gpt2-large", "batch_size": 20},
        {"name": "gpt2-xl", "dir_name": "gpt2-xl", "batch_size": 20},
        {"name": "mistralai/Mistral-7B-v0.1", "dir_name": "mistral", "batch_size": 20},  # Large model, use smaller batches
        {"name": "Qwen/Qwen2.5-3B", "dir_name": "qwen", "batch_size": 20},  # Medium model, use medium batches
    ]

    # Set up device
    cuda_available = torch.cuda.is_available()
    device_str = "cpu" # Default to CPU

    if args.devices is None or args.devices.lower() == "auto":
        if cuda_available:
            device_str = "cuda" # Use default CUDA device (usually cuda:0)
            print("CUDA available. Using default GPU.")
        else:
            print("CUDA not available. Using CPU.")
    else:
        try:
            device_ids = [int(d.strip()) for d in args.devices.split(',')]
            if not device_ids:
                 raise ValueError("Device list cannot be empty.")

            if cuda_available:
                # Use the first specified device ID
                target_device_id = device_ids[0]
                if target_device_id < 0 or target_device_id >= torch.cuda.device_count():
                     raise ValueError(f"Specified device ID {target_device_id} is invalid. Available devices: {list(range(torch.cuda.device_count()))}")
                device_str = f"cuda:{target_device_id}"
                print(f"CUDA available. Using specified GPU: {device_str}")
                if len(device_ids) > 1:
                    print("Warning: Multiple GPUs specified, but only the first one ({device_str}) will be used. Data parallelism is not implemented in this script.")
            else:
                 # CUDA specified but not available
                 print(f"Warning: CUDA devices ({args.devices}) specified, but CUDA is not available. Falling back to CPU.")
                 device_str = "cpu"

        except ValueError as e:
            print(f"Error parsing devices argument '{args.devices}': {e}. Falling back to CPU.")
            device_str = "cpu"

    device = torch.device(device_str)
    print(f"Selected device: {device}")

    # Create base output directory
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Process each prompt
    for prompt_idx, prompt in enumerate(args.prompts):
        print(f"\nProcessing prompt {prompt_idx+1}/{len(args.prompts)}: '{prompt}'")
        
        # Create prompt-specific directory
        prompt_dir_name = f"prompt_{prompt_idx+1}"
        prompt_dir = base_dir / prompt_dir_name
        prompt_dir.mkdir(exist_ok=True)
        
        # Save prompt to a text file for reference
        with open(prompt_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # Generate p values once for this prompt
        p = generate_p_values(args.max_length)
        
        # Process each model
        for model_info in models:
            print(f"\nGenerating with {model_info['name']} on device {device}...")
            model_dir = prompt_dir / model_info["dir_name"]
            model_dir.mkdir(exist_ok=True)
            
            # Determine batch size - use provided batch_size, or model-specific recommendation, or full batch
            batch_size = args.batch_size or model_info.get("batch_size")
            if batch_size:
                print(f"  Using batch size of {batch_size} for this model")
            
            # Initialize model
            model = Generator(model_name=model_info["name"])
            model = model.to(device)
            
            # Prepare for temperature sampling
            temp_temperature = generate_temperature(args.num_samples).to(device)
            temp_nucleus = torch.ones(args.num_samples).to(device)
            
            print(f"  Generating with temperature sampling...")
            temp_texts = model.generate_text(
                num_samples=args.num_samples,
                max_length=args.max_length,
                p=p,
                temperature=temp_temperature,
                nucleus=temp_nucleus,
                prompt=prompt,
                batch_size=batch_size
            )
            
            # Save temperature results
            temp_results = {
                "metadata": {
                    "model_name": model_info["name"],
                    "num_samples": args.num_samples,
                    "max_length": args.max_length,
                    "nucleus": False,
                    "temperatures": True,
                    "prompt": prompt,
                    "p_values": p.tolist()
                },
                "texts": [
                    {
                        "nucleus": float(temp_nucleus[i].cpu().item()),
                        "temperature": float(temp_temperature[i].cpu().item()),
                        "text": text
                    }
                    for i, text in enumerate(temp_texts)
                ]
            }
            
            temp_output_file = model_dir / f"{model_info['dir_name']}-temp.json"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(temp_results, f, ensure_ascii=False, indent=2)
            
            # Prepare for nucleus sampling
            nucleus_temperature = torch.ones(args.num_samples).to(device)
            nucleus_nucleus = generate_nucleus(args.num_samples).to(device)
            
            print(f"  Generating with nucleus sampling...")
            nucleus_texts = model.generate_text(
                num_samples=args.num_samples,
                max_length=args.max_length,
                p=p,
                temperature=nucleus_temperature,
                nucleus=nucleus_nucleus,
                prompt=prompt,
                batch_size=batch_size
            )
            
            # Save nucleus results
            nucleus_results = {
                "metadata": {
                    "model_name": model_info["name"],
                    "num_samples": args.num_samples,
                    "max_length": args.max_length,
                    "nucleus": True,
                    "temperatures": False,
                    "prompt": prompt,
                    "p_values": p.tolist()
                },
                "texts": [
                    {
                        "nucleus": float(nucleus_nucleus[i].cpu().item()),
                        "temperature": float(nucleus_temperature[i].cpu().item()),
                        "text": text
                    }
                    for i, text in enumerate(nucleus_texts)
                ]
            }
            
            nucleus_output_file = model_dir / f"{model_info['dir_name']}-nucleus.json"
            with open(nucleus_output_file, 'w', encoding='utf-8') as f:
                json.dump(nucleus_results, f, ensure_ascii=False, indent=2)
 
            # Explicitly free memory
            del model
            torch.cuda.empty_cache()

    print(f"\nAll generations completed. Results saved in {base_dir}")

if __name__ == "__main__":
    main() 