from abc import ABC, abstractmethod
import random
import copy
from enums import PromptType, GenerationStep, MCQ_STEP_COMBINATIONS, Enum

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
        question = data['prompt']
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
    GenerationStep.answer: "a single word or a short phrase that can be related to the previous step or steps",
    GenerationStep.question: "a one-sentence query based on the previous step or steps",
    GenerationStep.distractor: "a plausible but incorrect answer choice for a multiple-choice question",
    GenerationStep.fact: "a short sentence stating a verifiable truth",
    GenerationStep.answer_question: "a question and answer pair, where the question is one sentence and the answer is a single word or a short phrase",
    GenerationStep.choices: "four possible options for a multiple-choice question, labeled A, B, C, and D",
}

# acqd

INPUT_REQUIREMENTS = {
    ("aqd", GenerationStep.answer): ['category'], 
    ("aqd", GenerationStep.question): ['category', 'answer'],   
    ("aqd", GenerationStep.distractor): ['category', 'question', 'answer'],   
    ("fpd", GenerationStep.fact): ['category'],   
    ("fpd", GenerationStep.answer_question): ['category', 'fact'],   
    ("fpd", GenerationStep.distractor): ['category', 'question_answer'],   
    ("hp", GenerationStep.choices): ['category'],   
    ("hp", GenerationStep.answer_question): ['category', 'choices'],   
}

def build_prompt(step_combination_name, step, data):
    inputs_needed = INPUT_REQUIREMENTS.get((step_combination_name, step), [])
    inputs = {inp: data[inp] for inp in inputs_needed}
    output_description = DESCRIPTIONS.get(step, "output")

    prompt = f"I will give you the following inputs:\n"
    for inp_name, inp_value in inputs.items():
        display_name = "Category" if inp_name == "category" else inp_name.capitalize()
        prompt += f"- {display_name}: \"{inp_value}\"\n"

    if step == GenerationStep.answer:
        prompt += f"\nYour task is to generate a concise answer related to the given Category. "
        prompt += f"The answer should be {output_description} within the topic of \"{data['category']}\".\n" # Explicitly mention category
    else:
        prompt += f"\nYour task is to generate the {step.value}.  A {step.value} is {output_description}.\n"

    if(step.value == "distractor"):
        prompt += f"Generate three plausible distractors. Format your output as \"[distractor1], [distractor2], [distractor3]\"."
    else:
        prompt += f"Format your output as \"[{step.value}]\"."
    return prompt


class GenericMCQPrompt(ZeroShotPrompt):
    def __init__(self, step_combination_name, step):
        super().__init__()
        self.step_combination_name = step_combination_name
        self.step = step

    def create_prompt(self, data):
        return build_prompt(self.step_combination_name, self.step, data)

# --- End Refactored MCQ Generation Prompts ---

class PromptFactory:

    def __init__(self):
        # This map is no longer strictly necessary
        self.prompt_type_map = {
            PromptType.qg: QuestionGenerationVanilla,
            PromptType.qg_cot: QuestionGenerationCoT,
            PromptType.qg_fewshot: QuestionGenerationFewShot,
            PromptType.qg_selfcheck: QuestionGenerationCheckAnswer,
            PromptType.qa: QuestionAnsweringVanilla,
            PromptType.qa_selfcons: QuestionAnsweringVanilla,
            PromptType.category_generation: CategoryGenerationPrompt,
        }


    def get_prompt_for_step(self, step: GenerationStep, step_combination_name):
        return GenericMCQPrompt(step_combination_name, step)

    def get_prompt(self, prompt_type, step_combination_name="aqd"):
        if prompt_type in self.prompt_type_map:
            return self.prompt_type_map[prompt_type]()
        elif prompt_type == PromptType.tree_generation:
            first_step = MCQ_STEP_COMBINATIONS[step_combination_name][0] 
            return self.get_prompt_for_step(first_step, step_combination_name)
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")