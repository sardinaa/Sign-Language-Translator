import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Positional Encoding
# ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # shape: [1, max_len, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

# ----------------------------------------------------------------------------
# Multi-Head Attention
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out   = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head attention
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # Combine heads
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out(output)

# ----------------------------------------------------------------------------
# Feed-Forward Network
# ----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ----------------------------------------------------------------------------
# Encoder Layer
# ----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Self-attention block with residual connection and layer norm
        attn_output = self.self_attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(attn_output))
        # Feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src

# ----------------------------------------------------------------------------
# Transformer Encoder (Stack of Encoder Layers)
# ----------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src

# ----------------------------------------------------------------------------
# Simplified Transformer Classifier with Softmax at the End
# ----------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, num_layers=4,
                 num_heads=8, d_ff=1024, dropout=0.3):
        """
        Args:
            input_dim: Dimensionality of landmark features.
            num_classes: Number of output classes (words in your ASL vocabulary).
            d_model: Embedding dimension.
        """
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional boolean mask of shape [batch_size, seq_len] indicating valid tokens.
        Returns:
            probs: Tensor of shape [batch_size, num_classes] containing classification probabilities.
        """
        # Embed and add positional encoding (with scaling)
        x = self.embedding(x) * math.sqrt(self.d_model)  # [B, seq_len, d_model]
        x = self.positional_encoding(x)                    # [B, seq_len, d_model]
        x = self.encoder(x, mask)                          # [B, seq_len, d_model]

        # Pool over the sequence. If a mask is provided, do a masked average.
        if mask is not None:
            mask = mask.unsqueeze(-1).float()   # [B, seq_len, 1]
            x = x * mask                        # Zero out padded elements
            x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, d_model]
        else:
            x, _ = torch.max(x, dim=1)

        x = self.dropout(x)
        logits = self.fc(x)                     # [B, num_classes]
        # Apply softmax to obtain probabilities.
        return logits

