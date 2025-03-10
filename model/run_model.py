from data_loader import PromptCollator
from checkpoint_handler import Checkpoint
from model_loader import ModelFactory
from enums import ModelType, PromptType, GenerationStep
from prompt import PromptFactory, ZeroShotPrompt
import pickle
import datasets
import tqdm
import os
import copy
import argparse
import itertools
import json
from datasets import Dataset, Features, Value, Sequence
import datetime
import random


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
    default="aqd",
    )
    parser.add_argument(
        "--create_hf_dataset",
        action='store_true',
        help="Create a Hugging Face dataset from the generated JSON file.",
        default=False,
    )
    parser.add_argument(
        "--hf_dataset_output_dir",
        type=str,
        help="Output directory for the Hugging Face dataset",
        default="/fs/clip-projects/rlab/atrey/qgqa/QG-vs-QA/script_results/hf_dataset_output",
    )
    parser.add_argument(
        "--hf_dataset_push_to_hub",
        action='store_true',
        help="Push the generated HF dataset to the Hugging Face Hub",
        default=False,
    )

    parser.add_argument(
        "--hf_repo_name",
        type=str,
        help="HF repo name to push",
        default="atreydesai/qgqa_generated",
    )

    parser.add_argument(
        "--use_choices_prompt",
        action='store_true',
        help="Use the choices-only prompt for answering generation.",
        default=False
    )
    parser.add_argument(
        "--bloom_level",
        type=str,
        help="Bloom's Taxonomy level for question generation (knowledge, comprehension, application)",
        choices=["knowledge", "comprehension", "application"],
        default=None,
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
        self.step_map = {
            'a': GenerationStep.answer,
            'q': GenerationStep.question,
            'd': GenerationStep.distractor,
            'f': GenerationStep.fact,
            'h': GenerationStep.choices,
            'p': GenerationStep.answer_question
        }


    def generate_mcq(self, initial_data, step_combination_name):
        data = copy.deepcopy(initial_data)
        generated_text = {}

        for i in range(len(step_combination_name)):
            current_step_char = step_combination_name[i]
            current_step = self.step_map.get(current_step_char)

            print(f"\n--- Step: {current_step.value} ---")
            print(f"Current data before prompt creation: {data}")

            prompt_class = self.prompt_factory.get_prompt_for_step(step_combination_name, i)
            if not prompt_class:
                print(f"Warning: No prompt template found for step: {current_step.value}")
                continue

            # Pass bloom_level to build_prompt if it's the question step
            if current_step == GenerationStep.question:
                prompt = prompt_class.create_prompt(data, bloom_level=self.args.bloom_level)  # Pass bloom_level here

            else:
                prompt = prompt_class.create_prompt(data)

            if prompt is None:
                print(f"Warning: No prompt generated for step: {current_step.value}")
                continue

            print(f"Prompt sent to LLM:\n{prompt}")
            out_text = self.model.generate_text(prompt)
            print(f"Raw output from LLM: {out_text}")
            print(f"Raw output from LLM (with repr()): {repr(out_text)}")

            cleaned_output = out_text.strip().replace("```json", "").replace("```", "")
            cleaned_output = cleaned_output.strip()

            print(f"Cleaned output: {cleaned_output}")

            try:
                parsed_output = json.loads(cleaned_output)
                if current_step.value in parsed_output:
                     generated_text[current_step.value] = parsed_output[current_step.value]
                     data[current_step.value] = parsed_output[current_step.value]
                else:
                    print(f"Warning: Expected key '{current_step.value}' not found in JSON.")
                    generated_text[current_step.value] = None
                    data[current_step.value] = None
            except json.JSONDecodeError as e:
                print(f"Warning: Output is not valid JSON.  Output: {cleaned_output}. Error: {e}")
                generated_text[current_step.value] = None
                data[current_step.value] = None
            print(f"Data after step {current_step.value}: {data}")

        return generated_text

def convert_json_to_hf_dataset(json_file_path, output_dir, step_combination, model_nickname, bloom_level):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_names = []
    questions = []
    distractors = []
    correct_answers = []
    categories = []
    prompts = []
    choices_only_prompts = []
    facts = []  # New: Fact
    answer_questions = [] #new: answer_question pair
    choices = []

    for i in range(len(data["raw_text"])):
        raw_text_data = data["raw_text"][i]

        if raw_text_data is None:  # Handle cases where raw_text_data could be None
          dataset_names.append(f"generated_{step_combination}")
          questions.append("")
          distractors.append([])
          correct_answers.append("")
          categories.append("")
          prompts.append("")
          choices_only_prompts.append("")
          facts.append("")
          answer_questions.append("")
          choices.append([]) #Append empty list, not the string "[]"
          continue

        question = raw_text_data.get("question", "")
        answer = raw_text_data.get("answer", "")
        distractor_string = raw_text_data.get("distractor", "")
        distractor_list = distractor_string.split(", ") if distractor_string else []
        fact = raw_text_data.get("fact", "") #Fact
        answer_question = raw_text_data.get("answer_question", "")#Answer Question
        choice_list = raw_text_data.get("choices", [])


        dataset_names.append(f"generated_{step_combination}")
        questions.append(question)
        distractors.append(distractor_list)
        correct_answers.append(answer)
        categories.append(data["prompt"][i]["category"])
        facts.append(fact) #Fact
        answer_questions.append(answer_question) #AQ
        choices.append(choice_list)

        if question and answer:
            # Only create prompts/choice prompts if both q and a exists
            choice_selection = [answer] + distractor_list
            random.shuffle(choice_selection)
            labeled_choices = [f"{chr(65 + j)}. {choice}" for j, choice in enumerate(choice_selection)]

            full_prompt = f"{question}\n" + "\n".join(labeled_choices)
            prompts.append(full_prompt)

            choices_only_prompt = "\n".join(labeled_choices)
            choices_only_prompts.append(choices_only_prompt)
        else:  # If no q or no a, don't create prompt
            prompts.append("")
            choices_only_prompts.append("")



    dataset_dict = {
        "dataset": dataset_names,
        "question": questions,
        "distractors": distractors,
        "correct_answer": correct_answers,
        "category": categories,
        "prompt": prompts,
        "choices_only_prompt": choices_only_prompts,
        "fact" : facts,  # New: Fact
        "answer_question" : answer_questions, #new
        "choices": choices, #new
    }

    features = Features({
        "dataset": Value("string"),
        "question": Value("string"),
        "distractors": Sequence(Value("string")),
        "correct_answer": Value("string"),
        "category": Value("string"),
        "prompt": Value("string"),
        "choices_only_prompt": Value("string"),
        "fact": Value("string"), #fact
        "answer_question": Value("string"),
        "choices": Sequence(Value("string")),
    })

    hf_dataset = Dataset.from_dict(dataset_dict, features=features)
    #MODIFIED SECTION
    final_output_dir = os.path.join(output_dir, f"hf_{model_nickname}", step_combination)
    if bloom_level:  # Add Bloom's level if provided
        final_output_dir = os.path.join(final_output_dir, bloom_level)
    os.makedirs(final_output_dir, exist_ok=True)

    hf_dataset.save_to_disk(final_output_dir)
    #END MODIFIED SECTION
    print(f"Dataset saved to {final_output_dir}")
    return final_output_dir


def main(args):
    # load model
    model_factory = ModelFactory()
    model = model_factory.get_model(args)

    # load checkpoints
    checkpoint_loader = Checkpoint(args) #Pass args here
    prompt_collator = PromptCollator(args)

    for prompt_type in args.prompt_types:
        for pt in prompt_type:
            checkpoint_loader.set_directories(pt)
            prompts = list(prompt_collator.get_prompts(pt, checkpoint_loader)) #prompts should be a list
            start, end = checkpoint_loader.setup_partition(len(prompts))

            outputs = checkpoint_loader.load_checkpoint()
            start += len(outputs['raw_text'])

            if pt == PromptType.tree_generation:
                generation_pipeline = GenerationPipeline(model, args)
                for idx in tqdm.tqdm(range(start, end)):
                    data_item = prompts[idx]
                    print(f"Data item type: {type(data_item)}")
                    print(f"Data item content: {data_item}")

                    if data_item is None:
                        outputs['raw_text'].append(None)
                        outputs['prompt'].append(None)
                        continue

                    initial_data = data_item
                    print("\n\n\n Initial data BEFORE generate_mcq:", initial_data)
                    generated_mcq = generation_pipeline.generate_mcq(initial_data, args.step_combination)
                    outputs['raw_text'].append(generated_mcq)
                    outputs['prompt'].append(data_item)
                    checkpoint_loader.save_checkpoint(outputs, False)
            elif pt == PromptType.answering_generation:
              for idx in tqdm.tqdm(range(start, end)):
                  prompt = prompts[idx]
                  if prompt is None:
                      outputs['raw_text'].append(None)
                      outputs['prompt'].append(None)
                      continue

                  prompt_data = {'prompt' : prompt}
                  out_text = model.generate_text(prompt)
                  outputs['raw_text'].append(out_text)
                  outputs['prompt'].append(prompt)
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
            checkpoint_loader.save_checkpoint(outputs, True)

            # Create HF dataset if requested
            if args.create_hf_dataset and pt == PromptType.tree_generation:
                json_file_path = checkpoint_loader.get_final_dir()
                print(f"Converting JSON file to HF dataset: {json_file_path}")
                try:
                    saved_dir = convert_json_to_hf_dataset(json_file_path, args.hf_dataset_output_dir, args.step_combination, args.model_nickname, args.bloom_level)

                    if args.hf_dataset_push_to_hub:
                        from datasets import load_from_disk
                        loaded_dataset = load_from_disk(saved_dir)
                        print("pushing to hub: ", args.hf_repo_name)
                        loaded_dataset.push_to_hub(args.hf_repo_name)

                except Exception as e:
                    print(f"Error during dataset conversion or pushing: {e}")

            elif args.create_hf_dataset:
                print("Warning! --create_hf_dataset currently only supported for tree_generation.  Skipping.")




if __name__ == '__main__':
    args = setup()
    main(args)