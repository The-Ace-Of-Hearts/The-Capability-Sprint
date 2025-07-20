#!/usr/bin/env python
"""
Capability Sprint – fine‑tune DistilBERT on IMDB.

Quick usage
-----------
# dry‑run (load + evaluate only)
python sprint_finetune.py --dry_run --subset 2000 --batch 8

# debug run on a 2 000‑row subset
python sprint_finetune.py --epochs 1 --subset 2000 --batch 8

# full one‑epoch train on all 25 k rows
python sprint_finetune.py --epochs 1 --batch 8
"""

import argparse
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from evaluate import load as load_metric
import inspect

# ░░░ argparse ░░░─────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='distilbert-base-uncased')
    ap.add_argument('--dataset', default='imdb')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--subset', type=int, default=0,
                    help="If >0, limit *training* rows to this many (quick test)")
    ap.add_argument('--dry_run', action='store_true',
                    help="Skip training; just load model & evaluate")
    return ap.parse_args()

# ░░░ main ░░░─────────────────────────────────────────────────────────────
def main():
    # ─── setup ───────────────────────────────────────────────────────────
    # Disable wandb to avoid prompts in non-interactive environments
    os.environ["WANDB_DISABLED"] = "true"

    args = parse_args()

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    raw_ds = load_dataset(args.dataset) # 25k train / 25k test

    # Tokenize the dataset
    def tok_fn(batch):
        return tokenizer(batch['text'], truncation=True)

    tokenized = raw_ds.map(tok_fn, batched=True, remove_columns=['text'])

    # Load metric
    metric = load_metric('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    )

    # ─── data splits ─────────────────────────────────────────────────────
    train_ds = tokenized['train'].shuffle(seed=42)
    if args.subset > 0:
        train_ds = train_ds.select(range(args.subset))

    # Always use a fixed subset for evaluation for consistent comparison
    eval_ds  = tokenized['test'].shuffle(seed=42).select(range(2000))

    # Initialize DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ─── dry‑run path ────────────────────────────────────────────────────
    if args.dry_run:
        print("Running dry run (evaluation only)...")
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        baseline = trainer.evaluate(eval_dataset=eval_ds)
        print("Dry‑run accuracy:", baseline["eval_accuracy"])
        return

    # ─── training setup ──────────────────────────────────────────────────
    # Note: Using 'eval_strategy' instead of 'evaluation_strategy'
    # as a workaround for a previous TypeError encountered in this environment.
    training_args = TrainingArguments(
        output_dir='./outputs',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy='epoch', # Using eval_strategy as requested
        save_strategy='epoch',
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # ─── diagnostics before training ────────────────────────────────────
    print(f"Dataset sizes: train={len(raw_ds['train'])}, test={len(raw_ds['test'])}")
    print(f"Model parameters: {model.num_parameters():,}")

    print("Running baseline evaluation before training...")
    baseline = trainer.evaluate(eval_dataset=eval_ds)
    print("Baseline accuracy:", baseline["eval_accuracy"])

    # ─── train ───────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train()

    # ─── final eval & report ─────────────────────────────────────────────
    print("Running final evaluation after training...")
    final_metrics = trainer.evaluate(eval_dataset=eval_ds)

    # Write report
    report_path = "report.txt"
    with open(report_path, "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Report written to {report_path}")

    # Save the trained model
    save_path = f"./outputs/model_epoch{args.epochs}"
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")

# ░░░ entry point ░░░──────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
