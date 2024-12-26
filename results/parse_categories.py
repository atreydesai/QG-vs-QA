import argparse
import json
from enum import Enum

class PromptType(Enum):
    qg = 'qg'
    qg_cot = 'qg_cot'
    qg_fewshot = 'qg_fewshot'
    qg_selfcheck = 'qg_selfcheck'

    qa = 'qa'
    qa_selfcons = 'qa_selfcons'
    category_generation = "category_generation"

class ModelType(Enum):
    hf_chat = 'hf_chat'
    open_ai = 'open_ai'
    cohere = 'cohere'
    anthropic = 'anthropic'

def enum_type(enum):
    enum_members = {e.name: e for e in enum}

    def converter(input):
        out = []
        for x in input.split():
            if x in enum_members:
                out.append(enum_members[x])
            else:
                raise argparse.ArgumentTypeError(f"You used {x}, but value must be one of {', '.join(enum_members.keys())}")
        return out

    return converter

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        help="String to identify this run",
        default="",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Directory where results are stored",
        default="./",
    )
    parser.add_argument(
        '--prompt_types', 
        nargs='*', 
        type=enum_type(PromptType), 
        help='Prompt types/experiments to run', 
        default=[]
    )
    args = parser.parse_args()
    return args

def parse_category(txt):
    if txt is None:
        return None 
    lines = txt.split('\n')
    for line in lines:
        if line.startswith("Category:"):
            return line[len("Category:"):].strip()
    return None

def main(args):
    run_name = args.run_name
    model_name = args.model_name
    res_dir = args.res_dir
    pt = args.prompt_types[0][0] 

    f = f'{res_dir}/{model_name}/{run_name}/{pt.value}.json'
    with open(f, 'r') as handle:
        data = json.load(handle)
    raw_out = data['raw_text']

    parsed_categories = []
    for out in raw_out:
        pc = parse_category(out)
        if pc is None:
            continue
        parsed_categories.append(pc)

    f = f'{res_dir}/{model_name}/{run_name}/{pt.value}+category.json'
    data['category'] = parsed_categories
    with open(f, 'w') as handle:
        json.dump(data, handle, indent=4)

if __name__ == '__main__':
    args = setup()
    main(args)