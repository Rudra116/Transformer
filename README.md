<div align="center">

# Transformer from Scratch

### A PyTorch implementation of *"Attention Is All You Need"*
### Vaswani et al., 2017 — faithfully reproduced, component by component

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-blueviolet?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/1706.03762)

---

> *"You don't understand something until you build it yourself."*

This repository is a **clean, from-scratch PyTorch implementation** of the Transformer architecture — the model behind GPT, BERT, T5, and virtually every modern language model. Built for **English → Italian translation** using the OPUS Books dataset, with support for Colab training, Beam Search decoding, and attention visualization.

</div>

---

## Architecture Overview

The Transformer replaces recurrence entirely with self-attention, allowing full parallelization and long-range dependency modeling.

```
Input Tokens
     │
     ▼
┌─────────────────┐
│ Input Embedding  │  ← Token IDs → d_model=512 dimensional vectors
│  × √(d_model)   │    Scaled by √512 as in the paper
└────────┬────────┘
         │
┌────────▼────────┐
│  Positional     │  ← sin/cos encoding injected into embeddings
│   Encoding      │    PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
└────────┬────────┘    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
         │
         ▼
┌──────────────────────────────────────────┐
│            ENCODER  (N=6 blocks)          │
│  ┌────────────────────────────────────┐  │
│  │  Multi-Head Self-Attention (h=8)   │  │
│  │  + Add & Layer Norm               │  │
│  ├────────────────────────────────────┤  │
│  │  Feed-Forward (d_ff=2048)          │  │
│  │  + Add & Layer Norm               │  │
│  └────────────────────────────────────┘  │
│              × 6 layers                   │
└──────────────────────┬───────────────────┘
                       │ encoder_output
         ┌─────────────┘
         ▼
┌──────────────────────────────────────────┐
│            DECODER  (N=6 blocks)          │
│  ┌────────────────────────────────────┐  │
│  │  Masked Multi-Head Self-Attention  │  │  ← prevents future token leakage
│  │  + Add & Layer Norm               │  │
│  ├────────────────────────────────────┤  │
│  │  Cross-Attention (Q←tgt, K/V←enc) │  │  ← attends to encoder memory
│  │  + Add & Layer Norm               │  │
│  ├────────────────────────────────────┤  │
│  │  Feed-Forward (d_ff=2048)          │  │
│  │  + Add & Layer Norm               │  │
│  └────────────────────────────────────┘  │
│              × 6 layers                   │
└──────────────────────┬───────────────────┘
                       │
┌──────────────────────▼───────────────────┐
│         Linear Projection + Softmax       │  ← (batch, seq_len, vocab_size)
└──────────────────────────────────────────┘
                       │
                  Output Token
```

---

## 🔬 Components Implemented

Every component from the paper is implemented in [`model.py`](model.py):

| Component | Class | Description |
|---|---|---|
| Input Embeddings | `InputEmbeddings` | Token ID → 512-dim vector, scaled by √d_model |
| Positional Encoding | `PositionalEncoding` | Sinusoidal PE added to embeddings; registered as buffer |
| Layer Normalization | `LayerNormalization` | Custom implementation with learnable α and β |
| Feed-Forward Block | `FeedForwardBlock` | Two linear layers with ReLU: d_model → 2048 → d_model |
| Multi-Head Attention | `MultiHeadAttentionBlock` | Scaled dot-product attention across h=8 heads |
| Residual Connection | `ResidualConnection` | Pre-norm + dropout before sublayer, then add |
| Encoder Block | `EncoderBlock` | Self-attention + FFN with two residual connections |
| Decoder Block | `DecoderBlock` | Masked self-attn + cross-attn + FFN with three skip connections |
| Full Encoder | `Encoder` | N=6 stacked encoder blocks + final layer norm |
| Full Decoder | `Decoder` | N=6 stacked decoder blocks + final layer norm |
| Projection Layer | `ProjectionLayer` | Linear layer mapping d_model → vocab_size |
| Transformer | `Transformer` | Full encoder-decoder model with `encode`, `decode`, `project` |

---

## ✨ Multi-Head Attention — Deep Dive

The heart of the Transformer. For each head, attention is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In code (`model.py`):

```python
@staticmethod
def attention(query, key, value, mask, dropout):
    d_k = query.shape[-1]
    # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attention_scores.masked_fill_(mask == 0, -1e9)  # causal mask in decoder

    attention_scores = attention_scores.softmax(dim=-1)
    return (attention_scores @ value), attention_scores
```

The 8 heads are computed in parallel by reshaping the query/key/value tensors:

```
(batch, seq_len, d_model)
    → (batch, seq_len, h, d_k)
    → (batch, h, seq_len, d_k)   ← each head sees d_k=64 dimensions
```

Outputs are concatenated and projected through W_o back to d_model.

---

## 📁 Project Structure

```
Transformer/
│
├── model.py              # Full Transformer architecture (all components)
├── dataset.py            # Tokenization, BilingualDataset, causal masking
├── train.py              # Training loop, checkpointing, validation
├── train_wb.py           # Training with Weights & Biases logging
├── translate.py          # Greedy decoding inference script
├── config.py             # Hyperparameters & file path helpers
│
├── Colab_Train.ipynb     # Google Colab training notebook (GPU)
├── Local_Train.ipynb     # Local training notebook
├── Inference.ipynb       # Run inference and inspect translations
├── Beam_Search.ipynb     # Beam search decoding implementation
├── attention_visual.ipynb # Visualize attention weights across heads
│
└── requirements.txt      # Dependencies
```

---

## Configuration

All hyperparameters live in [`config.py`](config.py):

```python
{
    "batch_size":   8,
    "num_epochs":   20,
    "lr":           1e-4,
    "seq_len":      350,
    "d_model":      512,       # embedding dimension
    "datasource":   "opus_books",
    "lang_src":     "en",
    "lang_tgt":     "it",      # English → Italian
    "model_folder": "weights",
    "preload":      "latest",  # resume from latest checkpoint
    "experiment_name": "runs/tmodel"
}
```

These match the original paper's defaults (d_model=512, h=8, N=6, d_ff=2048, dropout=0.1).

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Rudra116/Transformer.git
cd Transformer
pip install -r requirements.txt
```

### 2. Train Locally

```bash
python train.py
```

Or open [`Local_Train.ipynb`](Local_Train.ipynb) in Jupyter.

### 3. Train on Google Colab (Recommended)

Open [`Colab_Train.ipynb`](Colab_Train.ipynb) — it handles dataset download, tokenizer building, and GPU training automatically.

### 4. Run Inference

```bash
python translate.py
```

Or use [`Inference.ipynb`](Inference.ipynb) for interactive translation.

### 5. Beam Search Decoding

Open [`Beam_Search.ipynb`](Beam_Search.ipynb) to run beam search and compare output quality against greedy decoding.

### 6. Visualize Attention

Open [`attention_visual.ipynb`](attention_visual.ipynb) to see what each attention head focuses on across encoder and decoder layers.

---

## 🔢 Model Hyperparameters (Paper Defaults)

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 512 | Embedding & hidden dimension |
| `N` | 6 | Number of encoder/decoder layers |
| `h` | 8 | Number of attention heads |
| `d_k = d_v` | 64 | Dimension per head (512 / 8) |
| `d_ff` | 2048 | Feed-forward inner dimension |
| `dropout` | 0.1 | Applied after attention & FFN |
| `seq_len` | 350 | Max sequence length |
| `batch_size` | 8 | Training batch size |
| `lr` | 1e-4 | Adam learning rate |

---

## 📊 Training

The model trains on the [OPUS Books](https://huggingface.co/datasets/opus_books) English-Italian corpus via HuggingFace datasets. A WordLevel tokenizer is trained from scratch on the data.

Key training features:
- **Checkpointing** — saves model weights each epoch; can resume from `"latest"`
- **Validation** — computes BLEU score and greedy-decoded samples during training
- **W&B Logging** — use `train_wb.py` for experiment tracking with Weights & Biases
- **Xavier Initialization** — all parameters with `dim > 1` initialized with `nn.init.xavier_uniform_`

---

## 📖 Reference Paper

> **Attention Is All You Need**
> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
> NeurIPS 2017 · [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

This implementation follows the paper closely — same architecture, same hyperparameters, same positional encoding formulation, same weight initialization strategy.

---

## Requirements

```
torch>=2.0
torchtext
datasets
tokenizers
torchmetrics
wandb          # optional, for W&B logging
tqdm
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🙋 Author

**Rudra116** · [github.com/Rudra116](https://github.com/Rudra116)

---

<div align="center">

*Built to understand the architecture that changed everything in NLP.*

</div>
