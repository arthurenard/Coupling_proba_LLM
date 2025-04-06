uv run generate.py \
--model_name "gpt2-xl" \
--num_samples 100 \
--max_length 30 \
--devices 1 \
--use_nucleus \
--prompt "It is unbelievable that" \
--output_json "gpt2_nucleus.json"