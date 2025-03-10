from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("atreydesai/mmlu_arc_categories", split="test")

# Replace with the actual path to your category_generation.json file
with open("/fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/script_results/gpt-4o-mini/default/category_generation.json", "r") as f:
    categories_data = json.load(f)
    raw_categories = categories_data["raw_text"]

if len(raw_categories) != len(dataset):
    raise ValueError(
        f"Number of categories ({len(raw_categories)}) does not match the number of rows in the dataset ({len(dataset)})."
    )


extracted_categories = []
for cat_string in raw_categories:
    # Split the string by "Category: " and take the second part.
    # Handle cases where "Category: " might not be present.
    parts = cat_string.split("Category: ")
    if len(parts) > 1:
        category = parts[1]
    else:
        #  Handle cases without "Category: "  -  use the whole string, or a default, or skip.
        #  Here, we use the entire string as the category.  Adjust as needed.
        category = cat_string
    extracted_categories.append(category)

def add_category(example, idx):
    example["category"] = extracted_categories[idx]
    return example

dataset_with_categories = dataset.map(add_category, with_indices=True)
dataset_with_categories.push_to_hub("atreydesai/mmlu_arc_categories", split="test")