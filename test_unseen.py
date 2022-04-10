# Load t5_model.pt form local directory
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration
import torch
from transformers import AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.load_state_dict(torch.load("t5_model_peregrine.pt", map_location="cpu"))

tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hypothesis = "A person dressed in a dress with flowers and a stuffed bee attached to it, is pushing a baby stroller down the street."
premise = "A person outside pushing a stroller."

input_sentence = '<s>' + hypothesis +  '</s>' + premise + '</s> '

input_ids_list = []
attention_mask_list = []

temp_encoding = tokenizer.encode_plus(input_sentence, 
            input_sentence,
            padding="longest", 
            return_token_type_ids=True, 
            max_length=256,
            )


input_ids = torch.Tensor(temp_encoding.input_ids)
attention_mask = torch.Tensor(temp_encoding.attention_mask)

input_ids = input_ids.to(device).long()
attention_mask = attention_mask.to(device).long()

# Set input_ids to be of shape (batch_size, seq_len)
input_ids = input_ids.unsqueeze(0) 
attention_mask = attention_mask.unsqueeze(0)

# create new list of inputs and attention_mask
input_ids_list = []
attention_mask_list = [] 

model.eval()
output = model.generate(
    input_ids = input_ids,
    attention_mask = attention_mask,
    num_beams = 10,
    max_length = 100,
    repetition_penalty = 2.0,
    length_penalty = 1.0,
    early_stopping = True,
    use_cache = True,
)
    

print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))