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
        frac_of_data=0.0004, # Set the fraction of data to be used for training (right now is very low, so it will only use 4 batches)
    ):
        # Load dataset and turn it to a pandas dataframe 
        dataset = load_dataset("esnli")
        self.train_df = pd.DataFrame.from_dict(dataset["train"])
        self.val_df = pd.DataFrame.from_dict(dataset["validation"])
        self.test_df = pd.DataFrame.from_dict(dataset["test"])
        
        # Refromat all data objects
        self.reformat_data()
        
        if frac_of_data < 1.0:
            # randomly sample frac_of_data of the data
            self.train_df = self.train_df.sample(frac=frac_of_data, replace=False, random_state=42)
            self.val_df = self.val_df.sample(frac=frac_of_data, replace=False, random_state=42)
            self.test_df = self.test_df.sample(frac=frac_of_data, replace=False, random_state=42)
        
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.initialize_data()

    def initialize_data(self):
        """ Initialize the data.
        """
        
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)
        self.test_data = self.load_data(self.test_df)
        
        
    def get_data_loaders(self, batch_size=64, shuffle=True):
        """ Initializes the data loaders, shuffles the samples if desired and puts everyhing in batches.

        Args:
            batch_size (int, optional): Batch size in the dataset. Defaults to 64.
            shuffle (bool, optional): Indicates if the dataset need to be shuffled. Defaults to True.

        Returns:
            DataLoaders: Returns the data loaders.
        """
        
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
        """ Loads the data into a TensorDataset. It takes the dataframe, tokenizes the input and labels and puts them in a tensor.

        Args:
            df (Pandas dataframe): The dataframe to be loaded.

        Returns:
           TensorDataset: The converted tensordataset
        """
        input_ids = []
        attention_mask = []
        token_type_ids = []
        target_ids = []
        
        premise_list = df["premise"].tolist()
        hypothesis_list = df["hypothesis"].tolist()
        explanation_list = df["explanation_1"].tolist()
        
        # Loop through all lists simultaniously
        for (
            premise, 
            hypothesis, 
            explanation
            ) in zip(
                premise_list, 
                hypothesis_list, 
                explanation_list
                ):
            # Concatenate premise and hypothesis with separator token :: and add an end of line token
            premise_hypothesis = f"{premise} :: {hypothesis} </s>"
            
            # Tokenize the premise and hypothesis
            hypothesis_premise_tokens = self.tokenizer.encode_plus(
                premise_hypothesis,
                truncation=True, 
                return_token_type_ids=True, 
                max_length=256,
                )
            
            # Tokenize the target explanation
            target_encoding = self.tokenizer.encode_plus(
                explanation,
                truncation=True,
                padding="longest",
                return_token_type_ids=True,
                max_length=256,
            )
            
            # For the input reurn the input_ids, attention_mask and token_type_ids as tensor to the list
            token_type_ids.append(torch.Tensor(hypothesis_premise_tokens.token_type_ids))
            attention_mask.append(torch.Tensor(hypothesis_premise_tokens.attention_mask))
            input_ids.append(torch.Tensor(hypothesis_premise_tokens.input_ids))
            
            # For the target return the target_ids as tensor to the list
            target_ids.append(torch.Tensor(target_encoding.input_ids))
        
        # Pad the sequences such that they are all of equal length. 
        # Batch_first means that we use the batch dimension as the first dimension.
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        input_ids = pad_sequence(input_ids, batch_first=True)
        target = pad_sequence(target_ids, batch_first=True)
        
        # Convert the tensors to tensordataset
        dataset = TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            target
        )
        return dataset

    def reformat_data(self):
        """ This will set every object in the dataset to be a list of strings. Except the label, which will stay an Int. 
        """

        self.test_df[
            [
                "premise", 
                "hypothesis", 
                "explanation_1", 
                "explanation_2", 
                "explanation_3"
            ]
        ] = self.train_df[
            [
                "premise", 
                "hypothesis", 
                "explanation_1", 
                "explanation_2", 
                "explanation_3"
                ]
            ].astype(
                str
                )
        
        self.val_df[
            [
                "premise",
                "hypothesis",
                "explanation_1",
                "explanation_2",
                "explanation_3"
                ]
        ] = self.val_df[
            [
                "premise", 
                "hypothesis",
                "explanation_1", 
                "explanation_2", 
                "explanation_3"]
            ].astype(
                str
                )   
        
        self.train_df[
            [
                "premise",
                "hypothesis", 
                "explanation_1"
                ]
        ] = self.train_df[
            [
                "premise", 
                "hypothesis", 
                "explanation_1"
                ]
            ].astype(
                str
                )