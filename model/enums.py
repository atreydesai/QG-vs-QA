from enum import Enum

class PromptType(Enum):
    qg = 'qg'
    qg_cot = 'qg_cot'
    qg_fewshot = 'qg_fewshot'
    qg_selfcheck = 'qg_selfcheck'
    qa = 'qa'
    qa_selfcons = 'qa_selfcons'
    category_generation = 'category_generation'
    tree_generation = 'tree_generation'

class GenerationStep(Enum):
    answer = 'answer'
    question = 'question'
    distractor = 'distractor'
    fact = 'fact'
    answer_question = 'answer_question'
    choices = 'choices'

class ModelType(Enum):
    hf_chat = 'hf_chat'
    open_ai = 'open_ai'
    cohere = 'cohere'
    anthropic = 'anthropic'


MCQ_STEP_COMBINATIONS = {
    "aqd": [GenerationStep.answer, GenerationStep.question, GenerationStep.distractor],
    "fpd": [GenerationStep.fact, GenerationStep.answer_question, GenerationStep.distractor],
    "hp": [GenerationStep.choices, GenerationStep.answer_question],
}