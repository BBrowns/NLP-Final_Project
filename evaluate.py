from transformers import T5ForConditionalGeneration
import torch
from transformers import AutoTokenizer
from dataloader import esnli
import pickle

import t5_trainer
import pandas as pd

from datasets import load_dataset

model = T5ForConditionalGeneration.from_pretrained("t5-base")
# get current working directory

model.load_state_dict(torch.load("t5_model_peregrine.pt", map_location="cpu"))

tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = esnli()

# load test loader 
train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size=32)

predictions, actuals = t5_trainer.evaluate(model, test_loader, tokenizer, device)
val_df = pd.DataFrame({"Predictions": predictions, "Target": actuals})

# load the esnli dataset
temp_df = load_dataset("esnli")

# take half of the test set 
test_df = pd.DataFrame.from_dict(temp_df["test"])
# get same length of tes_df
test_df = test_df.iloc[:len(val_df)]

# concatenate the two datasets on same row 
final_df = pd.concat([test_df, val_df], axis=1)


final_df.to_csv(f"predictions/evaluation.csv", sep= ";")

# write predictions to a pickle file
with open(f"predictions/evaluation.pkl", "wb") as f:
    pickle.dump(predictions, f)