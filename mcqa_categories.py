from datasets import load_dataset, DatasetDict, Dataset
import random
HF_TOKEN = "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"

###############################
#GENERAL INFORMATION ##########
#Features in 'train' split: {'dataset': Value(dtype='string', id=None), 'question': Value(dtype='string', id=None), 'choices': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'answer_letter': Value(dtype='string', id=None)}
#
###############################


ds_nishant = load_dataset("nbalepur/mcqa_artifacts", cache_dir="/fs/clip-scratch/atrey/huggingface")
mmlu_entries = [entry for entry in ds_nishant['test'] if entry['dataset'].startswith("mmlu")]
arc_entries = [entry for entry in ds_nishant['test'] if entry['dataset'] == "ARC"]

random.seed(2024)  # for reproducibility
mmlu_sample = random.sample(mmlu_entries, min(5, len(mmlu_entries)))
arc_sample = random.sample(arc_entries, min(5, len(arc_entries)))

def restructure_entry(entry):
    correct_answer_index = ord(entry['answer_letter']) - ord('A')
    correct_answer = entry['choices'][correct_answer_index]
    distractors = [choice for i, choice in enumerate(entry['choices']) if i != correct_answer_index]
    dataset_name = 'MMLU' if entry['dataset'].startswith('mmlu') else 'ARC'
    return {
        "dataset": dataset_name,
        "question": entry['question'],
        "distractors": distractors,
        "correct_answer": correct_answer,
        "category": ""
    }

combined_data = [restructure_entry(entry) for entry in (mmlu_sample + arc_sample)]
final_dataset = Dataset.from_dict({key: [entry[key] for entry in combined_data] for key in combined_data[0].keys()})
final_dataset = DatasetDict({"test": final_dataset})


print(final_dataset['test'][0])
"""
{
    'question': 'The population of country X ', 
    'distractors': ['the people ', 'the people of country Y', 'the country X.'], 
    'correct_answer': 'standard of living twice as much as country Y.', 
    'category': ''
}
"""


final_dataset.push_to_hub("atreydesai/mmlu_arc_categories",token=HF_TOKEN)

