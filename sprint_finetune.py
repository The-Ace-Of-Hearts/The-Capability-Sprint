%%writefile The-Capability-Sprint/sprint_finetune.py
#!/usr/bin/env python
"""
Capability Sprint – fine‑tune DistilBERT on IMDB.

Quick usage
-----------
python sprint_finetune.py --dry_run --subset 2000 --batch 8
python sprint_finetune.py --epochs 1 --subset 2000 --batch 8
python sprint_finetune.py --epochs 1 --batch 8
"""

import argparse, os, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from evaluate import load as load_metric

# ░ argparse ░─────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='distilbert-base-uncased')
    ap.add_argument('--dataset', default='imdb')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--subset', type=int, default=0,
                    help="If >0, limit training rows to this many")
    ap.add_argument('--dry_run', action='store_true',
                    help="Skip training; just load + evaluate")
    return ap.parse_args()

# ░ main ░────────────────────────────────────────────
def main():
    os.environ["WANDB_DISABLED"] = "true"
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    raw_ds = load_dataset(args.dataset)

    def tok_fn(batch): return tokenizer(batch['text'], truncation=True)
    tokenized = raw_ds.map(tok_fn, batched=True, remove_columns=['text'])

    metric = load_metric('accuracy')
    def compute_metrics(ep):
        logits, labels = ep
        return metric.compute(
            predictions=np.argmax(logits, axis=-1), references=labels
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    )

    train_ds = tokenized['train'].shuffle(seed=42)
    if args.subset > 0:
        train_ds = train_ds.select(range(args.subset))
    eval_ds  = tokenized['test'].shuffle(seed=42).select(range(2000))

    if args.dry_run:
        baseline = Trainer(
            model=model,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        ).evaluate(eval_dataset=eval_ds)
        print("Dry‑run accuracy:", baseline["eval_accuracy"])
        return

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
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print(f"Dataset sizes: train={len(raw_ds['train'])}, test={len(raw_ds['test'])}")
    print(f"Model parameters: {model.num_parameters():,}")
    print("Baseline accuracy:", trainer.evaluate()["eval_accuracy"])

    trainer.train()

    final_metrics = trainer.evaluate()
    with open("report.txt", "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")
    print("Report written to report.txt")

    trainer.save_model("./outputs/model_epoch{}".format(args.epochs))

if __name__ == '__main__':
    main()