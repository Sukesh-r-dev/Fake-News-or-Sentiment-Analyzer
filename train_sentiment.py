# train_sentiment.py

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate

def main():
    print("1. Loading dataset...")
    # Load a small sample of the IMDB dataset for fast experimentation
    dataset = load_dataset("imdb", split="train[:2000]")
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print("2. Tokenizing data...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    print("3. Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define evaluation metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./sentiment_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    print("4. Starting fine-tuning (this may take a few minutes)...")
    trainer.train()

    print("5. Saving model and testing inference...")
    model.save_pretrained("./my_sentiment_model")
    tokenizer.save_pretrained("./my_sentiment_model")

    # Test the pipeline
    classifier = pipeline("sentiment-analysis", model="./my_sentiment_model", tokenizer=tokenizer)
    sample_text = "This product exceeded all my expectations! Absolutely worth every penny."
    result = classifier(sample_text)
    
    print(f"\nTest Text: '{sample_text}'")
    print(f"Prediction Result: {result}")

if __name__ == "__main__":
    main()
