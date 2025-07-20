\# Capability Sprint – Fine‑tuning DistilBERT on IMDB



\## Environment



```json

{

  "accelerate": "1.8.1",

  "datasets": "2.14.4",

  "torch": "2.6.0+cu124",

  "transformers": "4.53.2"

}

## Baseline metrics (20 Jul 2025)
Dataset sizes: train=25000, test=25000
Model parameters: 66,955,010
Baseline accuracy: 0.4455

## Fine‑tune results (20 Jul 2025, 1 epoch)
Validation loss: 0.229  
Validation accuracy: 0.924  
Training runtime: 6 min (batch 8, fp16)

## Fine‑tune results (20 Jul 2025, 1 epoch)
Validation loss: 0.228  
Validation accuracy: 0.922  
Evaluation runtime: 7 s (batch 8, fp16)


