from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np

def tokenize_function(examples):
    return tokenizer(examples["explanation_1"], padding='max_length', truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = load_dataset("esnli")

max_len = len(max(dataset['train']['explanation_1']))
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", model_max_length=max_len)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Selected smaller datasets to test out if it works
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Uncomment below and comment above to do training on entire dataset
# train_dataset = tokenized_datasets["train"]
# eval_dataset = tokenized_datasets["test"]

model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# I set the batch sizes a little higher so that finetuning wouldn't take decades
training_args = TrainingArguments(output_dir="test_trainer", per_device_train_batch_size=100, per_device_eval_batch_size=100)

metric = load_metric("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))
# Uncomment below and comment above to do prediction on entire validation dataset
# validation_dataset = tokenized_datasets["validation"]
print(trainer.predict(validation_dataset).metrics)

# Could save the model by uncommenting this, might be huge though
# trainer.save_model()
