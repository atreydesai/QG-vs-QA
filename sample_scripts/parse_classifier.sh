# name of the model (identified by the API)
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# run to extract
run_name="default"
# experiment to extract
experiments=("category_generation")
experiments_str=$(IFS=" "; echo "${experiments[*]}")
# results directory
res_dir="/fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/script_results"

python3 /fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/results/parse_categories.py \
--run_name="$run_name" \
--model_name="$model_name" \
--prompt_types="$experiments_str" \
--res_dir="$res_dir"