from enum import Enum
from enums import PromptType
import os
import datasets
import numpy as np
from prompt import PromptFactory
import json
import contextlib
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
HF_TOKEN = "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"


from abc import ABC, abstractmethod

class DataFetcher(ABC):
    @abstractmethod
    def get_data(self):
        """Retrieve data from the source."""
        pass

class EntityFetcher(DataFetcher):

    def __init__(self, ds_name, split_name, run_num_only=False):
        self.ds = self.load_hf_dataset(ds_name)
        if type(self.ds) == datasets.dataset_dict.DatasetDict:
            if split_name in self.ds.keys():
                self.ds = self.ds[split_name]
            else:
                raise ValueError(f"The split does not exist in your dataset dictionary: {split_name}")
        if run_num_only:
            self.ds = self.ds.filter(lambda ex: 'num' in ex['category'])

    def load_hf_dataset(self, ds_name):
        if os.path.isfile(ds_name):
            ds = datasets.load_from_disk(ds_name)
        else:
            ds = datasets.load_dataset(ds_name)
        return ds
            
    def get_data(self, column_name='answer'):
        return list(self.ds[column_name])
    
    

class QuestionFetcher(DataFetcher):

    def __init__(self, ds_dir, split_name = None):

        if split_name == None:
            if os.path.isfile(ds_dir):
                with open(ds_dir, 'r') as handle:
                    self.ds = json.load(handle)
            else:
                raise ValueError(f"The question file does not exist: {ds_dir}")
        else:
            self.ds = self.load_hf_dataset(ds_dir)[split_name]
            #return ds["question"]

    def load_hf_dataset(self, ds_name):
        if os.path.isfile(ds_name):
            ds = datasets.load_from_disk(ds_name)
        else:
            ds = datasets.load_dataset(ds_name)
        return ds

    def get_data(self, column_name='question'):
        return self.ds[column_name]
    

class MCQADatasetFetcher(DataFetcher):

    def __init__(self, dataset_name):
        self.ds = datasets.load_dataset(dataset_name, token=HF_TOKEN)["test"] 

    def get_data(self, column_name='question'):
        return list(self.ds[column_name])
    
    
class DataFetcherFactory:

    @staticmethod
    def get_data_fetcher(prompt_type, args, checkpoint_loader):
        if prompt_type in {PromptType.qg, PromptType.qg_cot, PromptType.qg_fewshot, PromptType.qg_selfcheck}:
            return EntityFetcher(args.dataset_name, args.inference_split)
        elif prompt_type in {PromptType.qa}:
            return QuestionFetcher(args.dataset_name, args.inference_split)
        elif prompt_type in {PromptType.qa_selfcons}:
            swapped_dir = checkpoint_loader.get_final_dir().replace('qa_selfcons', 'qg').replace('.json', '+question.json')
            return QuestionFetcher(swapped_dir)
        elif prompt_type == PromptType.category_generation:  
            return MCQADatasetFetcher(args.dataset_name)
        elif prompt_type ==PromptType.tree_generation:
            return MCQADatasetFetcher(args.dataset_name)
        else:
            raise ValueError(f"Unsupported DataFetcher type: {prompt_type}")

class PromptCollator:
    def __init__(self, args):
        self.prompt_factory = PromptFactory()
        self.data_fetcher_factory = DataFetcherFactory()
        self.args = args

    def get_prompts(self, prompt_type, checkpoint_loader):
        data_fetcher = self.data_fetcher_factory.get_data_fetcher(prompt_type, self.args, checkpoint_loader)
        prompt_parser = self.prompt_factory.get_prompt(prompt_type)

        #yield prompts one by one.
        for p in data_fetcher.get_data():
            prompt = None if p is None else prompt_parser.create_prompt({'input': p})
            yield prompt
        