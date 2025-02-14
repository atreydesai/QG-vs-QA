from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("atreydesai/mmlu_arc_categories", split="test")

with open("/fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/script_results/meta-llama/Meta-Llama-3-8B-Instruct/default/category_generation.json", "r") as f:  # Changed the filename to a placeholder.  Replace with actual filename
    categories_data = json.load(f)
    raw_categories = categories_data["raw_text"]

if len(raw_categories) != len(dataset):
    raise ValueError(
        f"Number of categories ({len(raw_categories)}) does not match the number of rows in the dataset ({len(dataset)})."
    )


extracted_categories = []
for cat_string in raw_categories:
    category = cat_string.split("Category: ")[1]
    extracted_categories.append(category)

def add_category(example, idx):
    example["category"] = extracted_categories[idx]
    return example

dataset_with_categories = dataset.map(add_category, with_indices=True)
dataset_with_categories.push_to_hub("atreydesai/mmlu_arc_categories", split="test")