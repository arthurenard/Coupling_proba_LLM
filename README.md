# GPT-2 Text Generation with PyTorch Lightning

This project implements a flexible GPT-2 text generation pipeline using PyTorch Lightning with various sampling strategies.

## Features

- Generate text samples using the GPT-2 model
- Control text generation using temperature and nucleus sampling
- Start generation from a custom prompt
- Save generated text and metadata to JSON
- GPU acceleration support

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Generate text samples using the provided script:

```bash
python generate.py [options]
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_name` | Name of the Hugging Face model to use | gpt2 |
| `--num_samples` | Number of text samples to generate | 4 |
| `--max_length` | Maximum length of the generated sequences | 50 |
| `--use_nucleus` | Enable nucleus sampling | False |
| `--use_temperature` | Enable temperature sampling | False |
| `--prompt` | Optional text prompt to start generation | None |
| `--devices` | Devices to use ('auto' or list of IDs like '0,1') | auto |
| `--output_json` | Path to save results as JSON | None |

### Examples

Generate 4 samples using default settings:
```bash
python generate.py
```

Generate 10 samples with temperature sampling:
```bash
python generate.py --num_samples 10 --use_temperature
```

Generate text continuing from a prompt:
```bash
python generate.py --prompt "The fox is" --max_length 100
```

Generate 20 samples with nucleus sampling and save to JSON:
```bash
python generate.py --num_samples 20 --use_nucleus --output_json results.json
```

## Technical Details

### Sampling Methods

- **Temperature Sampling**: Controls the randomness of predictions by scaling the logits before applying softmax. Higher temperature (> 1.0) increases randomness, lower temperature (< 1.0) makes the distribution more concentrated.

- **Nucleus Sampling**: Also known as top-p sampling, this approach samples from the smallest set of tokens whose cumulative probability exceeds a threshold p.

When combining both methods, the temperature is applied first, followed by nucleus filtering.

## JSON Output Format

When using `--output_json`, results are saved in the following format:

```json
{
  "metadata": {
    "model_name": "gpt2",
    "num_samples": 4,
    "max_length": 50,
    "nucleus": true,
    "temperatures": true,
    "prompt": "Optional prompt text"
  },
  "texts": [
    {
      "nucleus": 0.0,
      "temperature": 0.0,
      "text": "Generated text for sample 1"
    },
    {
      "nucleus": 0.33,
      "temperature": 0.4,
      "text": "Generated text for sample 2"
    }
  ]
}
```
```