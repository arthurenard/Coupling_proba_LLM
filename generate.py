########### Imports ###########
# Import libraries
import pytorch_lightning as pl
import argparse
import torch
import json
from pathlib import Path

# Import functions
from src.Generator import Generator
from src.generate_values import generate_p_values, generate_temperature, generate_nucleus


########### Parse arguments ###########
parser = argparse.ArgumentParser(description="Generate text using GPT-2 and PyTorch Lightning.")
parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the Hugging Face model to use.")
parser.add_argument("--num_samples", type=int, default=4, help="Number of text samples to generate.")
parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the generated sequences.")
parser.add_argument("--use_temperature", action="store_true", help="Use temperature sampling.")
parser.add_argument("--use_nucleus", action="store_true", help="Use nucleus sampling.")
parser.add_argument("--prompt", type=str, default=None, help="Optional text prompt to start generation.")
parser.add_argument("--devices", type=str, default="auto", help="Devices to use ('auto', or list of IDs like '0,1').")
parser.add_argument("--output_json", type=str, default=None, help="Path to save results as JSON.")
parser.add_argument("--batch_size", type=int, default=None, 
                    help="Batch size for generation. Use for large models to avoid CUDA OOM errors.")
args = parser.parse_args()


########### Set up devices ###########
if args.devices.lower() == "auto":
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    try:
        devices = [int(d.strip()) for d in args.devices.split(',')]
        if len(devices) == 1:
            devices = devices[0] # Trainer expects int for single device
    except ValueError:
        raise ValueError(f"Error: Invalid device specified: {args.devices}. Use 'auto' or comma-separated integers.")


########### Display generation info ###########
print(f"Generating {args.num_samples} samples with max length {args.max_length} using {args.model_name}")
if args.prompt:
    print(f"Starting with prompt: '{args.prompt}'")


########### Prepare text generation ###########
num_samples = args.num_samples
model = Generator(model_name=args.model_name)

p = generate_p_values(args.max_length)

if args.use_temperature and args.use_nucleus:
    print("WARNING: Temperature and nucleus sampling are both set to True. This could lead to incoherent text generation.")
if not args.use_temperature and not args.use_nucleus:
    print("WARNING: Temperature and nucleus sampling are both set to False. This will lead to useless text generation.")
    num_samples = 1

if args.use_temperature:
    temperature = generate_temperature(num_samples)
else:
    temperature = torch.ones(num_samples)

if args.use_nucleus:
    nucleus = generate_nucleus(num_samples)
else:
    nucleus = torch.ones(num_samples)

########### Set up the model ###########
model = model.to(devices)
temperature = temperature.to(devices)
nucleus = nucleus.to(devices)

########### Generate text ###########
decoded_texts = model.generate_text(
    num_samples=num_samples,
    max_length=args.max_length,
    p=p,
    temperature=temperature,
    nucleus=nucleus,
    prompt=args.prompt,
    batch_size=args.batch_size
)


########### Display or save generated text ###########
if not args.output_json:
    print("\nGenerated Texts:")
    for i, text in enumerate(decoded_texts):
        print(f"--- Sample {i+1} at temperature {temperature[i]:.4f} and nucleus {nucleus[i]:.4f} ---")
        print(text)
else:
    # Create the results dictionary with the required structure
    results = {
        "metadata": {
            "model_name": args.model_name,
            "num_samples": num_samples,
            "max_length": args.max_length,
            "nucleus": args.use_nucleus,
            "temperatures": args.use_temperature,
            "prompt": args.prompt,
            "p_values": p.tolist()  # Convert tensor to list for JSON serialization
        },
        "texts": [
            {
                "nucleus": float(nucleus[i].cpu().item()),
                "temperature": float(temperature[i].cpu().item()),
                "text": text
            }
            for i, text in enumerate(decoded_texts)
        ]
    }
    
    # Ensure output directory exists
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output_json}")