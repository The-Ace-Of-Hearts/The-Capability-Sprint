#!/usr/bin/env python
"""
Capability Sprint – fine‑tune DistilBERT on IMDB.
Run:  python sprint_finetune.py --epochs 1
"""

import argparse, os, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from evaluate import load as load_metric

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

    training_args = TrainingArguments(
        output_dir='./outputs',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'].shuffle(seed=42).select(range(6000)),  # small subset for speed
        eval_dataset=tokenized['test'].select(range(2000)),
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(f"./model_epoch{args.epochs}")

if __name__ == '__main__':
    main()
