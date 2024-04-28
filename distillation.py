import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForCausalLM
import torch
from transformers import TrainingArguments, Trainer
from functools import partial

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model_name = "EleutherAI/pythia-160m"
filename = 'updated_clean_tmu.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# number of steps to train for
max_steps = 100
# number of examples in each batch
batch_size = 16
# number of epochs to train for
epochs = 1
# learning rate
learning_rate = 1.0e-5
# Save model to this direction
trained_model_name = f"pythia_steps"
output_dir = trained_model_name
save_dir = f'{output_dir}/final'

# Create a function that will be used to decode the model's predictions
def decode_predictions(model, examples):
    # Generate predictions
    inputs = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    
    # Decode the predicted tokens to strings
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return {"predicted_output": predictions}

def tokenize_function(examples):

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        examples['prompt'],
        padding="max_length",
        return_tensors="np",
        text_target=examples['labels'],
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

training_args = TrainingArguments(

    # Learning rate
    learning_rate=learning_rate,
    # Number of training epochs
    # num_train_epochs=epochs,
    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,
    # Batch size for training
    per_device_train_batch_size=batch_size,
    # Directory to save model checkpoints
    output_dir=output_dir,
    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=100, # Number of update steps between two evaluations
    save_steps=100, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1, # Batch size for evaluation
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=False,
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv(filename)

    # Create a new column for the prompts
    data['prompt'] = data.apply(lambda x: f"{x['source']} This sentence is wrong because {x['explanation']}, what is the corrected version of it? ", axis=1)
    # Use the corrected sentence as labels
    data['labels'] = data['output']    
    # Convert the DataFrame to a Hugging Face dataset
    dataset = Dataset.from_pandas(data)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )

    # Split the dataset into training, validation, and test sets
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
    train_validation_split = split_dataset["train"].train_test_split(test_size=0.1111, shuffle=True, seed=123)
    train_dataset = train_validation_split["train"]
    validation_dataset = train_validation_split["test"]
    test_dataset = split_dataset["test"]

    # Load the model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.to(device)

    # Create a Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # Train the model
    training_output = trainer.train()

    # Save the model
    trainer.save_model(save_dir)
    print("Saved model to:", save_dir)

    decode_predictions_with_model = partial(decode_predictions, base_model)
    test_dataset_with_predictions = test_dataset.map(decode_predictions_with_model, batched=True, batch_size=8)

    # Convert the results to a pandas DataFrame
    test_df = test_dataset_with_predictions.to_pandas()

    # Save the DataFrame to a CSV file
    test_df.to_csv('test_with_predictions.csv', index=False)

    print("Saved predictions to 'test_with_predictions.csv'")