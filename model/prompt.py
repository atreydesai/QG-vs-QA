from abc import ABC, abstractmethod
import random
import copy
from enums import PromptType, GenerationStep

# Abstract base class for implementing zero-shot prompts
class ZeroShotPrompt(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def create_prompt(self, data):
        """Create a zero-shot prompt"""
        pass

# Question Generation Prompts
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

# Question Answering Prompts
class QuestionAnsweringVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        question = data['input']
        prompt = f'Generate the answer to the question: "{question}". Give just the answer and no explanation. Please format your output as "Answer: [insert generated answer]". If no possible answer exists say "IDK".'
        return prompt

class CategoryGenerationPrompt(ZeroShotPrompt):
    def create_prompt(self, data):
        question = data['input']
        prompt = f"The following question is about: [QUESTION] {question} \nWhat is the most appropriate category for this question? \nCategory:"
        prompt = f"""
        The following question is about: [QUESTION] {question}\n
What is the most appropriate category for this question? I have provided 10 examples of questions and their associated categories\n

Examples:\n

1. Question: What is the formula for calculating the area of a circle?\n
Category: Mathematics - Geometry\n

2. Question: How does photosynthesis work in plants?\n
Category: Science - Biology\n

3. Question: What were the major themes in Shakespeare's Hamlet?\n
Category: Literature - Drama\n

4. Question: What are the steps to bake a chocolate cake?\n
Category: Cooking - Baking\n

5. Question: How do I troubleshoot a slow internet connection?\n
Category: Technology - Troubleshooting\n

6. Question: What are the current ethical considerations surrounding artificial intelligence?\n
Category: Ethics - Technology\n

7. Question: What is the best strategy for investing in stocks for long-term growth?\n
Category: Finance - Investing\n

8. Question: Who painted the Mona Lisa and during what period?\n
Category: Art - Renaissance\n

9. Question: What are the benefits of regular exercise on mental health?\n
Category: Health - Mental Wellness\n

10. Question: What was the primary cause of the American Civil War?\n
Category: History - American History\n

Now, classify this question:\n

[QUESTION] {question}\n
Category:\n
        """

        return prompt

class ConceptGenerationPrompt(ZeroShotPrompt):
     def create_prompt(self, data):
        prompt = f'Generate a concept related to the following: [Data] {data}. The concept should be a single word or a short phrase. Please format your output as "Concept: [insert generated concept]".'
        return prompt

class AnswerGenerationPrompt(ZeroShotPrompt):
    def create_prompt(self, data):
        concept = data.get('concept', '')  # Get concept from data, default to empty if not present
        prompt = f'Generate a concise answer related to the concept: "{concept}". Please format your output as "Answer: [insert generated answer]".'
        return prompt

class DistractorGenerationPrompt(ZeroShotPrompt):
    def create_prompt(self, data):
        question = data.get('question', '')
        answer = data.get('answer', '')
        prompt = f'Generate three plausible but incorrect answer choices (distractors) for the following question: "{question}" where the correct answer is "{answer}". Ensure distractors are of similar length and type as the answer. Format your output as "Distractor 1: [distractor1], Distractor 2: [distractor2], Distractor 3: [distractor3]".'
        return prompt
    
class FactGenerationPrompt(ZeroShotPrompt):
     def create_prompt(self, data):
        concept = data.get('concept', '')
        prompt = f'Generate a fact related to the concept: "{concept}". Please format your output as "Fact: [insert fact]".'
        return prompt
    
class ChoicesGenerationPrompt(ZeroShotPrompt):
    def create_prompt(self, data):
        concept = data.get('concept', '')
        prompt = f'Generate four possible choices for a multiple-choice question related to the concept: "{concept}". Ensure one choice is the correct answer, and the others are plausible distractors. Format your output as "Choice A: [choice_a], Choice B: [choice_b], Choice C: [choice_c], Choice D: [choice_d]". Indicate the correct answer with an asterisk (*).'
        return prompt




class PromptFactory:

    def __init__(self):

        self.prompt_type_map = {
            PromptType.qg: QuestionGenerationVanilla,
            PromptType.qg_cot: QuestionGenerationCoT,
            PromptType.qg_fewshot: QuestionGenerationFewShot,
            PromptType.qg_selfcheck: QuestionGenerationCheckAnswer,

            PromptType.qa: QuestionAnsweringVanilla,
            PromptType.qa_selfcons: QuestionAnsweringVanilla,

            PromptType.category_generation: CategoryGenerationPrompt,
        }
    
    def get_prompt_for_step(self, step: GenerationStep):
        prompt_map = {
            GenerationStep.concept: ConceptGenerationPrompt,
            GenerationStep.answer: AnswerGenerationPrompt,
            GenerationStep.question: QuestionGenerationVanilla,
            GenerationStep.distractor: DistractorGenerationPrompt,
            GenerationStep.fact: FactGenerationPrompt,
            GenerationStep.answer_question: QuestionAnsweringVanilla,
            GenerationStep.choices: ChoicesGenerationPrompt,
        }
        return prompt_map.get(step)

    def get_prompt(self, prompt_type):
        if prompt_type in self.prompt_type_map:
            return self.get_prompt_for_step[prompt_type]()
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")
