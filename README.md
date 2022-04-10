# NLP-Final_Project

## Installing dependencies 
The dependencies can be installed using pip install -r requirements.txt

## Running model 
The model can be run using python3 t5_trainer.py. t5_trainer.py will call the dataloader.py file, which will preprocess the data into the correct format and returns the dataloaders. 

In t5_trainer.py in the main, you can adjust the hyperparameters to your own wishes. 

## Evaluate data
The file evaluate.py can be used to extensively evaluate the data. It will call the model and predict output based on the test_loader. 

## Test on unseen data
To test the model on unseen data, run 'python3 test_unseen.py'. In the file you can add the the hypothesis and premise of your preference. 
Add 'map_location="cpu"' to torch.load if you're not using a gpu. 

## Evaluating the output



