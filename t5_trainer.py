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
    """Fine-tunes the T5 model. 

    Args:
        model : The model to be fine-tuned.
        train_loader (DataLoader): The train data loader.
        val_loader (DataLoader): The validation data loader.
        optimizer (torch.optim): The desired optimizer.
        wandb : The wandb logger.
    """
    # Start training (this is just like normal training with epochs only now its being tracked)
    for epoch in range(wandb.config.epochs):
        start = time.time()
        print("Epoch: ", epoch)
        # Set the model to train
        model.train()
        total_train_loss = 0 
        print("training loader:", len(train_loader))
        
        count = 0 
        # Iterate over the batches in the loader
        for batch, (
            token_type_ids, 
            attention_mask,
            input_ids,
            target_ids,
        ) in enumerate(train_loader):
            print("batch number:", count)
            # Set the optimizer to zero grad to avoid accumulating gradients
            optimizer.zero_grad()
            
            # Set all tensors to the device
            token_type_ids = token_type_ids.to(device).long()
            attention_mask = attention_mask.to(device).long()
            input_ids = input_ids.to(device).long()
            target_ids = target_ids.to(device).long()
            
            # Set the target ids and labels 
            y_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone().detach()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask= attention_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            
            # Get the loss
            loss = outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Add the loss to the total loss
            total_train_loss += loss.item()
            
            count+=1
 
        # Calculate the average loss
        train_loss = total_train_loss / len(train_loader)
        
        # Evaluate the model on the validation set
        model.eval()
        total_val_loss = 0
        
        # Avoid using the gradient for validation. Otherwise we are still training the model
        with torch.no_grad():
            for batch, (
                input_ids, 
                token_type_ids, 
                attention_mask,
                target_ids,
            ) in enumerate(val_loader):
                token_type_ids = token_type_ids.to(device).long()
                attention_mask = attention_mask.to(device).long()
                input_ids = input_ids.to(device).long()
                target_ids = target_ids.to(device).long()
                
                # Set the target ids and labels 
                y_ids = target_ids[:, :-1].contiguous()
                lm_labels = target_ids[:, 1:].clone().detach()
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask= attention_mask,
                    decoder_input_ids=y_ids,
                    labels=lm_labels,
                )
                
                # Get the loss
                loss = outputs[0]
                
                # Add the loss to the total loss
                total_val_loss += loss.item()

        # Calculate the average loss
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        # return time in hours and minutes
        hours, rem = divmod(end-start, 3600)

        # Log the training loss and validation loss
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
    """Evaluating the model on the test set.

    Args:
        model: The model to be used for evaluation.
        test_loader (DataLoader): The test data loader.
        wandb: The wandb logger.
    """
    model.eval()
    total_test_loss = 0 

    # Avoid using the gradient for evaluation. Otherwise we are still training the model
    with torch.no_grad():
        for batch, (
            token_type_ids, 
            attention_mask,
            input_ids,
            target_ids,
        ) in enumerate(test_loader):
            # Set all tensors to the device
            token_type_ids = token_type_ids.to(device).long()
            attention_mask = attention_mask.to(device).long()
            input_ids = input_ids.to(device).long()
            target_ids = target_ids.to(device).long()
            
            # Set the target ids and labels 
            y_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone().detach()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask= attention_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            
            # Get the loss
            loss = outputs[0]
            
            # Add the loss to the total loss
            total_test_loss += loss.item()
        
        # Calculate the average loss
        test_loss = total_test_loss / len(test_loader)
        
        # Log the test loss
        wandb.log(
            {
                "test_loss": test_loss,
                "test_size": len(test_loader)
                   }
            )
    
            

if __name__ == "__main__":
    
    # The weights and biases (wandb) libary let's us keep track of the model's hyperparameters and configurations during training. 
    # You might want to make your own account here: https://wandb.ai/
    # After that you can set project and entity according to your own settings.
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

    # Load the dataset from dataloader
    dataset = esnli()
    
    # Load dataloaders
    train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size = wandb.config.batch_size)
    
    # set model configuration and model
    config = AutoConfig.from_pretrained("t5-base", ) 
    model = T5ForConditionalGeneration.from_pretrained("t5-base", config=config)
    wandb.watch(model)
    
    # Use AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=wandb.config.lr)
    
    model.to(device)
        
    train_t5(model, train_loader, val_loader, optimizer, wandb)
    
    evaluate(model, test_loader, wandb)
    
    torch.save(model.state_dict(), "t5_model.pt")
    wandb.save("t5_model.pt")
    