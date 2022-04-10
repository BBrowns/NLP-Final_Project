# NLP-Final_Project

## Installing dependencies 
The dependencies can be installed using pip install -r requirements.txt

## Dataset 
The used dataset can be found here [https://huggingface.co/datasets/esnli] and [https://github.com/OanaMariaCamburu/e-SNLI]. You do not have to download it. It will get loaded directly by the script.

## Running model 
The model can be run using `python3 t5_trainer.py`. `t5_trainer.py` will call the `dataloader.py` file, which will preprocess the data into the correct format and returns the dataloaders. 

In `t5_trainer.py` in the main, you can adjust the hyperparameters to your own wishes. The trainer will output and save a model. 

## Evaluate data
The file `evaluate.py` can be used to extensively evaluate the data. It will call the model and predict output based on the test_loader. It can be run by `python3 evaluate.py`

## Test on unseen data
To test the model on unseen data, run `python3 test_unseen.py`. In the file you can add the the hypothesis and premise of your preference. 
Add `map_location="cpu"`to torch.load if you're not using a gpu. 

## Evaluating the output
Using the `predictions.csv` you can run the `Evaluation+error_analysis.ipynb` notebook after putting the `.csv` file into a `/contents/` folder. This wil display all metric scores. 


