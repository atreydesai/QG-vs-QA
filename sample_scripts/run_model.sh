#!/bin/bash

# activate your environment here!
conda activate qgqa

# dataset details
inference_split="test"
dataset_name="atreydesai/mmlu_arc_categories"

# name of the model (identified by the API)
# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# model_name="mistralai/Mistral-Nemo-Instruct-2407"
model_name="mistralai/Mistral-Nemo-Instruct-2407"

# model type (see enums.py). Currently supported: hf_chat (Huggingface), open_ai, cohere, anthropic
model_type="hf_chat"

# how to identify this run
run_name="default"

# API tokens
hf_token= # huggingface read token (for downloading gated models)
open_ai_token= # OpenAI token (for GPT models)
cohere_token=... # Cohere token (Command-R)
anthropic_token=... # Anthropic token (Claude)

# generation parameters
temperature=0.0
min_tokens=5
max_tokens=1000

device_map="auto" # device map ('cpu', 'cuda', 'auto')
partition="full"  # partition of the dataset. can be "full" or in halves (e.g. "first_half"), quarters (e.g. "first_quarter"), or eigths (e.g. "first_eighth")

# experiment to run
# see all possible experiments in: /mcqa-artifacts/model/data_loader.py
# experiments=("tree_generation")
experiments=("tree_generation")
step_combination="qad"
use_choices_only="false"
bloom_level="application"  #an be "knowledge", "comprehension", "application"


# directory setup
res_dir="/fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/script_results" # Results folder directory
cache_dir="/fs/clip-scratch/atrey" # Cache directory to save the model
hf_dataset_output_dir="/fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/script_results/hf_dataset_output" # Where to store hf dataset locally
hf_repo_name="atreydesai/qgqa_generated"  #HF hub repo name

experiments_str=$(IFS=" "; echo "${experiments[*]}")

# Create HF dataset after generation?
create_hf_dataset="true"  # Set to "true" or "false"
push_to_hub="true" #local storage or push to hub


echo "hi"
# add the correct file below
# there are also flags for `load_in_4bit` and `load_in_8bit`
python3 /fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/model/run_model.py \
--run_name="$run_name" \
--model_nickname="$model_name" \
--model_name="$model_name" \
--model_type="$model_type" \
--dataset_name="$dataset_name" \
--inference_split="$inference_split" \
--partition="$partition" \
--hf_token="$hf_token" \
--open_ai_token="$open_ai_token" \
--cohere_token="$cohere_token" \
--anthropic_token="$anthropic_token" \
--device_map="$device_map" \
--temperature="$temperature" \
--min_tokens="$min_tokens" \
--max_tokens="$max_tokens" \
--prompt_types="$experiments_str" \
--res_dir="$res_dir" \
--cache_dir="$cache_dir" \
--step_combination="$step_combination" \
$(if [ "$create_hf_dataset" = "true" ]; then echo "--create_hf_dataset"; fi) \
--hf_dataset_output_dir="$hf_dataset_output_dir" \
$(if [ "$push_to_hub" = "true" ]; then echo "--hf_dataset_push_to_hub"; fi) \
--hf_repo_name="$hf_repo_name" \
$(if [ "$use_choices_only" = "true" ]; then echo "--use_choices_prompt"; fi) \
$(if [ "$bloom_level" != "" ]; then echo "--bloom_level=$bloom_level"; fi)


echo "DONE"