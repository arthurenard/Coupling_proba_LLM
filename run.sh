uv run generate.py \
--model_name gpt2 \
--num_samples 100 \
--max_length 20 \
--devices 1 \
--use_temperature \
--prompt "It is unbelievable that" \
--output_json "results.json"