# Transformer Implementation in PyTorch

This repository contains an implementation of the Transformer model in PyTorch. The Transformer is a neural network architecture introduced in the paper "Attention is All You Need" by Vaswani et al. It is widely used for natural language processing tasks such as machine translation, text summarization, and more.

Features

Encoder-Decoder Structure: Includes both encoder and decoder modules as shown in the Transformer architecture.

Multi-Head Attention: Implements scaled dot-product multi-head attention.

Feed Forward Layers: Includes position-wise feed-forward networks.

Positional Encoding: Adds positional information to word embeddings.

Masking: Implements source and target sequence masking for autoregressive tasks.

Requirements

Python 3.8+

PyTorch 1.10+

CUDA (optional for GPU support)

Install the required Python packages with:

pip install torch

Usage

Model Definition

You can define the Transformer model using the following code snippet:

from transformer import Transformer

# Define parameters
src_vocab_size = 10000
trg_vocab_size = 10000
src_pad_idx = 0
trg_pad_idx = 0
embed_size = 256
num_layers = 6
heads = 8
forward_expansion = 4
dropout = 0.1
max_length = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the model
model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embed_size,
    num_layers,
    forward_expansion,
    heads,
    dropout,
    device,
    max_length,
).to(device)

Example Input and Output

You can use the Transformer model with dummy data as follows:

import torch

# Example inputs
batch_size = 32
src_seq_length = 20
trg_seq_length = 20

src = torch.randint(0, src_vocab_size, (batch_size, src_seq_length)).to(device)
trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_length)).to(device)

# Forward pass
out = model(src, trg[:, :-1])
print(out.shape)  # Output shape: (batch_size, trg_seq_length - 1, trg_vocab_size)

Training

You can integrate the model into your training pipeline. Use a loss function like nn.CrossEntropyLoss and an optimizer like Adam:

import torch.nn as nn
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

File Structure

transformer.py: Contains the Transformer model implementation.

README.md: Provides an overview of the project.

References

Vaswani, Ashish, et al. "Attention is All You Need." Advances in Neural Information Processing Systems, 2017.
