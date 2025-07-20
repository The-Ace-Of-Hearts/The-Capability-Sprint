#!/usr/bin/env python

"""
Capability Sprint – fine‑tune DistilBERT on IMDB.
Run:  python sprint_finetune.py --epochs 1
"""


import argparse, os, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding) # Import DataCollatorWithPadding
from evaluate import load as load_metric
import transformers # Import transformers here
import inspect # Import inspect here


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='distilbert-base-uncased')
    ap.add_argument('--dataset', default='imdb')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)
    return ap.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.dataset)
    def tokenize(ex): return tokenizer(ex['text'], truncation=True)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])
    metric = load_metric('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2)

    # Print transformers version inside the script
    print(f"Transformers version inside script: {transformers.__version__}")
    # Print the file path of TrainingArguments
    print(f"TrainingArguments defined in: {inspect.getsourcefile(TrainingArguments)}")

    # Initialize DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./outputs',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'].shuffle(seed=42),
        eval_dataset=tokenized['test'].shuffle(seed=42).select(range(2000)),
        compute_metrics=compute_metrics,
        data_collator=data_collator # Add data collator here
    )
    # --- KR‑1 diagnostics --------------------------------------
    print(f"Dataset sizes: train={len(tokenized['train'])}, test={len(tokenized['test'])}")
    print(f"Model parameters: {model.num_parameters():,}")

    #   Quick baseline eval (no training yet)
    baseline = trainer.evaluate()
    print("Baseline accuracy:", baseline["eval_accuracy"])
    # -----------------------------------------------------------
    trainer.train()
    trainer.save_model(f"./model_epoch{args.epochs}")

if __name__ == '__main__':
    main()
