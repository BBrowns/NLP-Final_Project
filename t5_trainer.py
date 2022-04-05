import enum
from torchmetrics import BLEUScore
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForSequenceClassification
from dataloader import esnli
import pandas as pd
import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from torch.optim import AdamW
import wandb
import time 



def train_t5(model, train_loader, val_loader, optimizer, wandb):
    
    for epoch in range(wandb.config.epochs):
        start = time.time()
        print("Epoch: ", epoch)
        model.train()
        total_train_loss = 0 
        for batch, in (
            token_type_ids, 
            attention_mask,
            input_ids,
            target_ids,
        ) in enumerate(train_loader):
            optimizer.zero_grad()
            token_type_ids = token_type_ids.to(device).long
            attention_mask = attention_mask.to(device).long
            input_ids = input.ids.to(device).long
            target_ids = target_ids.to(device).long
            
            y_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone.detach()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask= attention_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
    
        train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch, in (
                token_type_ids, 
                attention_mask,
                input_ids,
                target_ids,
            ) in enumerate(val_loader):
                token_type_ids = token_type_ids.to(device).long
                attention_mask = attention_mask.to(device).long
                input_ids = input_ids.to(device).long
                target_ids = target_ids.to(device).long
                
                y_ids = target_ids[:, :-1].contiguous()
                lm_labels = target_ids[:, 1:].clone.detach()
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask= attention_mask,
                    decoder_input_ids=y_ids,
                    labels=lm_labels,
                )
                
                loss = outputs[0]
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        # return time in hours and minutes
        hours, rem = divmod(end-start, 3600)

        wandb.log(
            {
                "val_loss": val_loss,
                "time": "{:0>2}:{:05.2f}".format(int(hours), rem),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_size": len(train_loader),
                "val_size": len(val_loader),
            }
        )
        
def evaluate(model, test_loader, wandb):
    model.eval()
    total_test_loss = 0 
    
    with torch.no_grad():
        for batch, (
            token_type_ids, 
            attention_mask,
            input_ids,
            target_ids,
        ) in enumerate(test_loader):
            token_type_ids = token_type_ids.to(device).long
            attention_mask = attention_mask.to(device).long
            input_ids = input_ids.to(device).long
            target_ids = target_ids.to(device).long
            
            y_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone.detach()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask= attention_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            
            loss = outputs[0]
            total_test_loss += loss.item()
        
        test_loss = total_test_loss / len(test_loader)
        wandb.log(
            {
                "test_loss": test_loss,
                "test_size": len(test_loader)
                   }
            )
    
            
                
        


if __name__ == "__main__":
    
    wandb.init(project="NLP-final-project", entity="baebrowns")
    wandb.config.epochs = 5
    wandb.config.batch_size = 64
    wandb.config.lr = 1e-5
    wandb.config.max_len = 512
    wandb.config.frac_of_data = 0.01
    wandb.config.multi_gpu = True
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    dataset = esnli()
    
    train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size = wandb.config.batch_size)
    
    # set model configuration and model
    config = AutoConfig.from_pretrained("t5-base") 
    model = T5ForConditionalGeneration.from_pretrained("t5-base", config=config)
    wandb.watch(model)
    

    
    optimizer = AdamW(model.parameters(), lr=wandb.config.lr)
    
    model.to(device)
    
    
    
    train_t5(model, train_loader, val_loader, optimizer, wandb)
    
    evaluate(model, test_loader, wandb)
    
    torch.save(model.state_dict(), "t5_model.pt")
    wandb.save("t5_model.pt")
    