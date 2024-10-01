import torch
from torch import nn
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
import math

import logging
import sys
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler
file_handler = logging.FileHandler('training_log.txt', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Starting the script")

# Load SNLI dataset
logger.info("Loading SNLI dataset")
dataset = load_dataset("snli", split="train[:1000]")  # Use 1000 examples
eval_dataset = load_dataset("snli", split="validation[:200]")  # Use 200 examples for evaluation
logger.info(f"Dataset loaded with {len(dataset)} training examples and {len(eval_dataset)} evaluation examples")

# Use GPT-2 tokenizer
logger.info("Loading GPT-2 tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

# Create a custom small GPT-2 style configuration
logger.info("Creating custom GPT-2 configuration")
custom_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=128,
    n_layer=4,
    n_head=4,
)
logger.info(f"Custom configuration: {custom_config}")

# Create a new model with random weights
logger.info("Initializing model with random weights")
model = GPT2LMHeadModel(config=custom_config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model moved to device: {device}")

# Print model size
model_size = sum(p.numel() for p in model.parameters())
logger.info(f"Model size: {model_size:,} parameters")

# Format the data for language modeling
logger.info("Formatting data for language modeling")


def format_text(example):
    return f"Premise: {example['premise']} Hypothesis: {example['hypothesis']} Label: {example['label']}"


# Apply formatting to the dataset
logger.info("Applying formatting to the dataset")
formatted_dataset = dataset.map(lambda x: {"text": format_text(x)})
formatted_eval_dataset = eval_dataset.map(lambda x: {"text": format_text(x)})
logger.info("Dataset formatting complete")

# Tokenize the dataset
logger.info("Tokenizing the dataset")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval_dataset = formatted_eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
logger.info("Dataset tokenization complete")
logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")

# Set up data collator
logger.info("Setting up data collator")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up training arguments
logger.info("Setting up training arguments")
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
)
logger.info(f"Training arguments: {training_args}")

# Create Trainer
logger.info("Creating Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)


# Custom logging callback
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(f"Step {state.global_step}: {logs}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f"Epoch {state.epoch} started")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {state.epoch} ended")


trainer.add_callback(LoggingCallback())

# Train the model
logger.info("Starting model training")
train_result = trainer.train()
logger.info("Model training completed")
logger.info(f"Training metrics: {train_result.metrics}")

# Evaluate the model
logger.info("Evaluating the model")
eval_result = trainer.evaluate()
perplexity = math.exp(eval_result["eval_loss"])
logger.info(f"Perplexity: {perplexity}")

# Save the fine-tuned model
logger.info("Saving the fine-tuned model")
trainer.save_model("./fine_tuned_custom_language_model")
logger.info("Model saved successfully")


# Function to generate text
def generate_text(prompt, max_length=100):
    logger.info(f"Generating text for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, temperature=0.7)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    return generated_text


# Test the model
logger.info("Testing the model")
test_prompt = "Premise: A person on a horse jumps over a broken down airplane. Hypothesis:"
print(generate_text(test_prompt))

logger.info("Script execution completed")