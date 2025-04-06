DIR="KV_gen"

# Run 3: Philosophical question
uv run big_generation.py \
--prompts "If a tree falls in a forest and no one is around to hear it," \
--output_dir "$DIR/tree_falls"

# Run 1: The answer to life, the universe and everything
uv run big_generation.py \
--prompts "The answer to life, the universe and everything is" \
--output_dir "$DIR/life_answer"

# Run 2: Strawberry counting
uv run big_generation.py \
--prompts "How many 'r' are in 'strawberry'?" \
--output_dir "$DIR/strawberry"

# Run 4: Chicken and egg
uv run big_generation.py \
--prompts "What came first, the chicken or the egg?" \
--output_dir "$DIR/chicken_egg"

# Run 5: Water taste
uv run big_generation.py \
--prompts "The taste of water is" \
--output_dir "$DIR/water_taste"

# Run 6: French idiom
uv run big_generation.py \
--prompts "Kill two birds with one stone in French is" \
--output_dir "$DIR/french_idiom"

# Run 7: Time question
uv run big_generation.py \
--prompts "What time is it?" \
--output_dir "$DIR/time_question"

# Run 8: Napoleon's horse
uv run big_generation.py \
--prompts "The color of the Napoleon white horse is" \
--output_dir "$DIR/napoleon_horse"

# Run 9: Fibonacci sequence
uv run big_generation.py \
--prompts "1, 1, 2, 3, 5, 8, 13," \
--output_dir "$DIR/fibonacci"
