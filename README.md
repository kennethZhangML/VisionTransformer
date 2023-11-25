# VisionTransformer
Implementations of the Vision Transformer, and accompanying pre-trained experiments

# Vision Transformer (ViT) - A Mathematical Explanation

## Introduction

The Vision Transformer (ViT) is a deep learning model architecture that has revolutionized computer vision tasks by applying the transformer architecture, originally designed for natural language processing, to image data. In this README, we will provide a mathematical explanation of the key components and operations of the Vision Transformer.

## Table of Contents

1. [Transformer Architecture](#transformer-architecture)
2. [Patch Embedding](#patch-embedding)
3. [Positional Encoding](#positional-encoding)
4. [Multi-Head Self-Attention](#multi-head-self-attention)
5. [Feedforward Neural Networks](#feedforward-neural-networks)
6. [ViT Architecture](#vit-architecture)

## Transformer Architecture

The Vision Transformer is based on the transformer architecture introduced in the "Attention is All You Need" paper by Vaswani et al. It consists of two main components: multi-head self-attention and feedforward neural networks. These components are applied recursively in multiple layers to process input data.

### Multi-Head Self-Attention

The core of the transformer architecture is the multi-head self-attention mechanism. Given an input sequence of embeddings, it calculates a weighted sum of all embeddings, with the weights determined by their relevance to each other.

The mathematical formulation of multi-head self-attention is as follows:

$$Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V$$


Where:
- `Q`, `K`, and `V` are query, key, and value matrices.
- `h` is the number of attention heads.
- `softmax` is the softmax activation function.
- `W_i` represents learnable weights for each head.

### Feedforward Neural Networks

After the attention mechanism, the transformer applies feedforward neural networks to each position's output independently. This introduces non-linearity and enables the model to capture complex patterns in the data.

The mathematical formulation of the feedforward neural network is as follows:

$$FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2$$


Where:
- `W_1` and `W_2` are learnable weight matrices.
- `ReLU` is the rectified linear unit activation function.

## Vision Transformer (ViT) Components

Now, let's delve into the specific components of the Vision Transformer (ViT) architecture.

### Patch Embedding

In ViT, the input image is divided into non-overlapping patches. Each patch is then linearly embedded into a lower-dimensional space using a trainable linear layer. The mathematical formulation is:

$$E(p_i) = x_i$$


Where:
- `x` is the input image.
- `E` is the embedding function.
- `p_i` represents patches.

### Positional Encoding

Since transformers do not have inherent notions of position, positional encodings are added to the patch embeddings to provide spatial information. ViT uses learned positional encodings, which are added element-wise to the patch embeddings.

The mathematical formulation is:

$$x_i = x_i + PE(p_i)$$

Where:
- `PE` is the positional encoding.
- `x` is the patch embeddings.
- `i` represents the position.

### ViT Architecture

The ViT architecture brings all these components together. It starts with patch embedding and positional encoding, followed by a stack of transformer blocks. These transformer blocks consist of multi-head self-attention and feedforward neural networks.

Finally, the output of the last transformer block is passed through a layer normalization step and then used for classification by a fully connected layer.



