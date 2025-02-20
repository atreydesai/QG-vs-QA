# imports and directory setup
from data_loader import PromptCollator
from checkpoint_handler import Checkpoint
from model_loader import ModelFactory
from enums import ModelType, PromptType, GenerationStep, MCQ_STEP_COMBINATIONS
from prompt import PromptFactory, ZeroShotPrompt
import pickle
import datasets
import tqdm
import os
import copy
import argparse
import itertools

# =========================================== Argument Setup ===========================================

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
        "--model_nickname",
        type=str,
        help="Nickname of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model on huggingface/OpenAI",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model_type",
        type=enum_type(ModelType),
        help="Type of the model: hf_chat",
        default="hf_chat",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Local path or huggingface path pointing to the dataset",
        default="nbalepur/QG_vs_QA_v2",
    )
    parser.add_argument(
        "--inference_split",
        type=str,
        help="Split of the dataset to use",
        default="full",
    )
    parser.add_argument(
        "--load_in_8bit",
        action='store_true',
        help="Should we load the model in 8 bit?",
        default=False,
    )
    parser.add_argument(
        "--load_in_4bit",
        action='store_true',
        help="Should we load the model in 4 bit?",
        default=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature of the model to use",
        default=0.7,
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        help="Minimum number of tokens to generate",
        default=5,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum number of tokens to generate",
        default=200,
    )
    parser.add_argument(
        "--device_map",
        type=str,
        help="Where to load the model ('cuda', 'auto', 'cpu')",
        default="auto",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Huggingface read token for access to models/datasets",
        default="",
    )
    parser.add_argument(
        "--open_ai_token",
        type=str,
        help="OpenAI token for access to the model",
        default="",
    )
    parser.add_argument(
        "--cohere_token",
        type=str,
        help="Cohere token for access to the model",
        default="",
    )
    parser.add_argument(
        "--anthropic_token",
        type=str,
        help="Anthropic token for access to the model",
        default="",
    )
    parser.add_argument(
        '--prompt_types',
        nargs='*',
        type=enum_type(PromptType),
        help='Prompt types/experiments to run',
        default=[]
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Which partition should be done",
        default="none",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        help="Absolute directory of the prompt folder",
        default="./",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Absolute directory of the cache folder for models",
        default="./",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Absolute directory of the output results folder",
        default="./",
    )

    parser.add_argument(
    "--step_combination",
    type=str,
    help="Step combination to use for MCQ generation",
    default="aqd",  # default combo changed to aqd
    )

    args = parser.parse_args()
    print(args)
    return args

# =========================================== Main Method ===========================================


class GenerationPipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.prompt_factory = PromptFactory()

    def generate_mcq(self, initial_data, step_combination_name):
        data = copy.deepcopy(initial_data)
        generated_text = {}
        # caqd => generated_text[category] = [...]
        steps = MCQ_STEP_COMBINATIONS.get(step_combination_name)
        if not steps:
            raise ValueError(f"Invalid step combination name: {step_combination_name}")

        for step in steps:
            print(f"\n--- Step: {step} ---")
            print(f"Current data before prompt creation: {data}") # Debugging print

            prompt_class = self.prompt_factory.get_prompt_for_step(step, step_combination_name)
            if not prompt_class:
                print(f"Warning: No prompt template found for step: {step}")
                continue

            prompt_template = prompt_class
            prompt = prompt_template.create_prompt(data)
            if prompt is None:
                print(f"Warning: No prompt generated for step: {step}")
                continue

            print(f"Prompt sent to LLM:\n{prompt}")

            out_text = self.model.generate_text(prompt)
            generated_text[step.value] = out_text
            data[step.value] = out_text
            print(f"Data after step {step}: {data}")

        return generated_text


def main(args):
    # load model
    model_factory = ModelFactory()
    model = model_factory.get_model(args)

    # load checkpoints
    checkpoint_loader = Checkpoint(args)
    prompt_collator = PromptCollator(args)

    for prompt_type in args.prompt_types:
        for pt in prompt_type:
            checkpoint_loader.set_directories(pt)
            prompts = list(prompt_collator.get_prompts(pt, checkpoint_loader)) # prompts needs to be a list
            start, end = checkpoint_loader.setup_partition(len(prompts))

            # Load current save state (optional, for resuming)
            outputs = checkpoint_loader.load_checkpoint()
            start += len(outputs['raw_text'])

            if pt == PromptType.tree_generation:
                # Initialize GenerationPipeline
                generation_pipeline = GenerationPipeline(model, args)
                # Iterate through prompts and generate MCQs
                for idx in tqdm.tqdm(range(start, end)):
                    data_item = prompts[idx] 
                    print(f"Data item type: {type(data_item)}") 
                    print(f"Data item content: {data_item}") 

                    if data_item is None:
                        outputs['raw_text'].append(None)
                        outputs['prompt'].append(None)
                        continue

                    # # Prepare initial data for GenerationPipeline - use data_item directly
                    initial_data = data_item  
                    print("\n\n\n Initial data BEFORE generate_mcq:", initial_data) 
                    generated_mcq = generation_pipeline.generate_mcq(initial_data, args.step_combination)
                    outputs['raw_text'].append(generated_mcq)
                    outputs['prompt'].append(data_item)
                    checkpoint_loader.save_checkpoint(outputs, False)

            else:
                for idx in tqdm.tqdm(range(start, end)):
                    prompt = prompts[idx] 

                    if prompt is None:
                        outputs['raw_text'].append(None)
                        outputs['prompt'].append(None)
                        continue
                    
                    prompt_data = {'prompt': prompt}
                    out_text = model.generate_text(prompt)

                    outputs['raw_text'].append(out_text)
                    outputs['prompt'].append(prompt)
                    checkpoint_loader.save_checkpoint(outputs, False)
            #final
            checkpoint_loader.save_checkpoint(outputs, True)

if __name__ == '__main__':
    args = setup()
    main(args)