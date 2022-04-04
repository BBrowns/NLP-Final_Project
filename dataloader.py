from lib2to3.pgen2 import token
from telnetlib import TELNET_PORT
from matplotlib import use
from pydantic import NonNegativeFloat
from spacy import explain, load
import torch

from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datasets import load_dataset

import pandas as pd

class esnli(Dataset):
    def __init__(
        self,
        frac_of_data=0.01,
    ):
        
        dataset = load_dataset("esnli")
        self.train_df = pd.DataFrame.from_dict(dataset["train"])
        self.val_df = pd.DataFrame.from_dict(dataset["validation"])
        self.test_df = pd.DataFrame.from_dict(dataset["test"])
        
        self.reformat_data()
        
        if frac_of_data < 1.0:
            # randomly sample frac_of_data of the data
            self.train_df = self.train_df.sample(frac=frac_of_data, replace=False, random_state=42)
            self.val_df = self.val_df.sample(frac=frac_of_data, replace=False, random_state=42)
            self.test_df = self.test_df.sample(frac=frac_of_data, replace=False, random_state=42)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.initialize_data()

    def initialize_data(self):
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)
        self.test_data = self.load_data(self.test_df)
        
        
    def get_data_loaders(self, batch_size=64, shuffle=True):
        print("initializing train data loader")
        train_loader = DataLoader(
            self.train_data, shuffle=shuffle, batch_size=batch_size
        )

        print("initializing val data loader")
        val_loader = DataLoader(self.val_data, shuffle=shuffle, batch_size=batch_size)

        print("initializing test data loader")
        test_loader = DataLoader(self.test_data, shuffle=shuffle, batch_size=batch_size)

        return train_loader, val_loader, test_loader
        
    def load_data(self, df):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        target_ids = []
        
        premise_list = df["premise"].tolist()
        hypothesis_list = df["hypothesis"].tolist()
        explanation_list = df["explanation_1"].tolist()
        
        for (premise, hypothesis, explanation) in zip(premise_list, hypothesis_list, explanation_list):
            # Concatenate premise and hypothesis with separator token </s>
            premise_hypothesis = f"{premise} </s> {hypothesis} </s>"
            
            hypothesis_premise_tokens = self.tokenizer.encode_plus(
                premise_hypothesis,
                truncation=True, 
                return_token_type_ids=True, 
                max_length=256,
                )
            
            target_encoding = self.tokenizer.encode_plus(
                explanation,
                truncation=True,
                padding="longest",
                return_token_type_ids=True,
                max_length=256,
            )
            
        
            token_type_ids.append(torch.Tensor(hypothesis_premise_tokens.token_type_ids))
            attention_mask.append(torch.Tensor(hypothesis_premise_tokens.attention_mask))
            input_ids.append(torch.Tensor(hypothesis_premise_tokens.input_ids))
            target_ids.append(torch.Tensor(target_encoding.input_ids))
        
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        input_ids = pad_sequence(input_ids, batch_first=True)
        target = pad_sequence(target_ids, batch_first=True)
        
        dataset = TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            target
        )
        return dataset

    def reformat_data(self):
       
        # for evey column in dataframe, train_df, set the datatype to str except for the label column
        self.test_df[["premise", "hypothesis", "explanation_1", "explanation_2", "explanation_3"]] = self.train_df[["premise", "hypothesis", "explanation_1", "explanation_2", "explanation_3"]].astype(str)
        self.val_df[["premise", "hypothesis", "explanation_1", "explanation_2", "explanation_3"]] = self.val_df[["premise", "hypothesis", "explanation_1", "explanation_2", "explanation_3"]].astype(str)   
        self.train_df[["premise", "hypothesis", "explanation_1"]] = self.train_df[["premise", "hypothesis", "explanation_1"]].astype(str)