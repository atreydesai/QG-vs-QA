from abc import ABC, abstractmethod
import random
import copy
from enums import PromptType, GenerationStep, Enum
import json

class ZeroShotPrompt(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def create_prompt(self, data):
        """Create a zero-shot prompt"""
        pass

class QuestionGenerationVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]". If no possible question exists say "IDK".'
        return prompt

class QuestionGenerationCoT(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Think step by step and reason before generating the question. After reasoning, please format your final output as "Question: [insert generated question]".'
        return prompt

class QuestionGenerationCheckAnswer(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]". After generating a question, answer your own question to verify that the answer is "{answer}", formatted as "Answer: [insert answer to generated question]".'
        return prompt

class QuestionGenerationFewShot(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = 'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]".'
        prompt += """

Answer: 328
Question: What is the sum of the first 15 prime numbers?

Answer: 710 survivors
Question: How many people survived the sinking of the RMS Titanic in 1912?

Answer: 648
Question: What is the product of 12 and 54?

Answer: 286 ayats
Question: How many verses are there in the longest chapter of the Quran, Surah Al-Baqarah?

Answer: 311
Question: What is the sum of the first three prime numbers greater than 100?

"""
        prompt += f"Answer: {answer}\nQuestion:"
        return prompt


class QuestionAnsweringVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        question = data['input']
        prompt = f'Generate the answer to the question: "{question}". Give just the answer and no explanation. Please format your output as "Answer: [insert generated answer]". If no possible answer exists say "IDK".'
        return prompt

class CategoryGenerationPrompt(ZeroShotPrompt):
    def create_prompt(self, data):
        question = data
        prompt = f"""Provide a category for this question:

Question: What is the formula for calculating the area of a circle?
Category: Mathematics - Geometry - Area of a Circle

Question: How does photosynthesis work in plants?
Category: Science - Biology - Photosynthesis

Question: What were the major themes in Shakespeare's Hamlet?
Category: Literature - Drama - Shakespeare's Hamlet

Question: What are the steps to bake a chocolate cake?
Category: Cooking - Baking - Baking a Cake

Question: How do I troubleshoot a slow internet connection?
Category: Technology - Troubleshooting - Internet Connection

Question: What are the current ethical considerations surrounding artificial intelligence?
Category: Ethics - Technology - Artificial Intelligence

Question: What is the best strategy for investing in stocks for long-term growth?
Category: Finance - Investing - Stocks

Question: Who painted the Mona Lisa and during what period?
Category: Art - Renaissance - Mona Lisa

Question: What are the benefits of regular exercise on mental health?
Category: Health - Mental Wellness - Exercise

Question: What was the primary cause of the American Civil War?
Category: History - American History - American Civil War

Question: {question}
Category:"""
        return prompt


# --- Refactored MCQ Generation Prompts ---

DESCRIPTIONS = {
    GenerationStep.answer: "a maximum of one sentence",
    GenerationStep.question: "a one-sentence query. This is just the question stem with no choices attached",
    GenerationStep.distractor: "three plausible-sounding answer choices for a multiple-choice question, seperated by commas",
    GenerationStep.fact: "a short sentence stating a verifiable truth",
    GenerationStep.answer_question: "a question and answer pair, where the question is one sentence and the answer is a single word or a short phrase",
    GenerationStep.choices: "four possible options for a multiple-choice question, labeled A, B, C, and D",
}

INPUT_DESCRIPTIONS = {
    'category': "the high-level category of the question",
    # Add descriptions for other input types here, if needed!
}

STEP_MAP = {
    'a': GenerationStep.answer,
    'q': GenerationStep.question,
    'd': GenerationStep.distractor,
    'f': GenerationStep.fact,
    'h': GenerationStep.choices,
    'p': GenerationStep.answer_question
}

BLOOM_DESCRIPTIONS = {
    "knowledge": "This will be of the level of thinking categorized as \"knowledge\" in Bloom's Taxonomy. This means rote factual knowledge of specific terminology, ways and means (i.e., conventions, trends, classifications and categories, criteria, methodology), universal axioms and/or abstractions accepted by the field or discipline (principles and generalizations, theories and structures).",
    "comprehension": "This will be of the level of thinking categorized as \"comprehension\" in Bloom's Taxonomy. This means understanding the meaning of information and materials by being able to translate materials from one form or format to another by explaining or summarizing and predicting consequences or effects.",
    "application": "This will be of the level of thinking categorized as \"application\" in Bloom's Taxonomy. This means using information and materials to solve new problems or respond to concrete situations that have a single or best answer through applying learned material such as rules, methods, concepts, principles, laws, and theories.",
}


def build_prompt(step_combination_name, current_step_index, data, bloom_level=None):
    current_step_char = step_combination_name[current_step_index]
    current_step = STEP_MAP.get(current_step_char)

    if current_step is None:
        raise ValueError(f"Invalid step character: {current_step_char}")

    # Determine required inputs
    required_inputs = {}
    required_inputs['category'] = data['category'] #category always required

    for i in range(current_step_index):
        prev_step_char = step_combination_name[i]
        prev_step = STEP_MAP[prev_step_char]
        required_inputs[prev_step.value] = data.get(prev_step.value) #prev steps

    #prompt :D
    prompt = "You are generating multiple choice questions which contain a question, an answer, and a set of three distractors. The answer is the correct answer to the question while the distractors are plausible-sounding answer choices.\n\n"
    prompt += f"Your current task is to generate {current_step.value} from the following inputs.  {current_step.value} is {DESCRIPTIONS[current_step]}.\n"



    if current_step == GenerationStep.question and bloom_level:
        prompt += BLOOM_DESCRIPTIONS.get(bloom_level, "") + "\n"


    for inp_name, inp_value in required_inputs.items():
        # print("\n\ninp_name is: " + inp_name)
        # print("inp_value is: " + inp_name + "\n\n")

        # inp_name corresponds to a GenerationStep enum
        try:
            generation_step = GenerationStep(inp_name)  # converting to enum
            if generation_step in DESCRIPTIONS:
                input_description = DESCRIPTIONS[generation_step]
                prompt += f"The input {inp_name} is {input_description}.\n"
            else:
                print(f"Warning: No description found for {inp_name} in DESCRIPTIONS")

        except ValueError:  # inp_name is NOT a GenerationStep
            if inp_name in INPUT_DESCRIPTIONS:
                prompt += f"The input {inp_name} is {INPUT_DESCRIPTIONS[inp_name]}.\n"
            else:
                print(f"Warning: No description found for input {inp_name} in INPUT_DESCRIPTIONS.  The prompt will proceed without a description for this input.")
                prompt += f"The input {inp_name} exists.\n"


    prompt += "The inputs are:\n"
    for inp_name, inp_value in required_inputs.items():
        prompt += f"{inp_name}: {inp_value}\n"

    prompt += f"\nPlease structure your output as a JSON with a key for {current_step.value}.\n"

    return prompt


class GenericMCQPrompt(ZeroShotPrompt):
    def __init__(self, step_combination_name, current_step_index):
        super().__init__()
        self.step_combination_name = step_combination_name
        self.current_step_index = current_step_index


    def create_prompt(self, data, bloom_level=None):
        return build_prompt(self.step_combination_name, self.current_step_index, data, bloom_level) 

# --- End Refactored MCQ Generation Prompts ---

class AnsweringGenerationPrompt(ZeroShotPrompt):
    def create_prompt(self, data):
        prompt_text = data['input']
        prompt = f"Answer the following question. {prompt_text}\nAnswer:"
        return prompt

class PromptFactory:

    def __init__(self):
        # This map is no longer necessary but idk if we need in the future so we keeping it
        self.prompt_type_map = {
            PromptType.qg: QuestionGenerationVanilla,
            PromptType.qg_cot: QuestionGenerationCoT,
            PromptType.qg_fewshot: QuestionGenerationFewShot,
            PromptType.qg_selfcheck: QuestionGenerationCheckAnswer,
            PromptType.qa: QuestionAnsweringVanilla,
            PromptType.qa_selfcons: QuestionAnsweringVanilla,
            PromptType.category_generation: CategoryGenerationPrompt,
            PromptType.answering_generation: AnsweringGenerationPrompt,
        }


    def get_prompt_for_step(self, step_combination_name, current_step_index):
        return GenericMCQPrompt(step_combination_name, current_step_index)

    def get_prompt(self, prompt_type, step_combination_name="aqd"):
        if prompt_type in self.prompt_type_map:
            return self.prompt_type_map[prompt_type]()
        elif prompt_type == PromptType.tree_generation:
            return self.get_prompt_for_step(step_combination_name, 0)
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")