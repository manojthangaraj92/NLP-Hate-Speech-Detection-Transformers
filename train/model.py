#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import random, math, sys
#from .utility import device

device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                     )

class SelfAttention(nn.Module):
    """
    Implementatation of simple self_attention, the heart of transformers
    """

    def __init__(self, embedding, heads=8, mask=False):
        """
        :param emb: embedding dimensions
        :param heads: number of heads that the input is divided
        :param mask: if masking needed, important in seq-seq models
        """

        super().__init__()

        assert embedding % heads == 0, f'Embedding dimension ({embedding}) should be divisible by number of heads ({heads})'

        self.embedding = embedding
        self.heads = heads
        self.mask = mask

        s = embedding // heads
        # - We will split the embedding into heads and feed them as inputs and concat at the end

        self.tokeys    = nn.Linear(embedding, embedding, bias=False)
        self.toqueries = nn.Linear(embedding, embedding, bias=False)
        self.tovalues  = nn.Linear(embedding, embedding, bias=False)

        self.unifyheads = nn.Linear(embedding, embedding)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.embedding, f'Input embedding dim ({e}) should match layer embedding dim ({self.embedding})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # We first compute the key/query/value's on the whole embedding vectors, and then split into the different heads.

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        #  get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, embedding, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, pos_embedding=None):
        super().__init__()

        self.attention = SelfAttention(embedding, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(embedding)
        self.norm2 = nn.LayerNorm(embedding)

        self.ff = nn.Sequential(

            nn.Linear(embedding, ff_hidden_mult * embedding),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embedding, embedding)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x


class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, embedding, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=embedding, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=embedding, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(embedding=embedding, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(embedding, num_classes)

        self.do = nn.Dropout(dropout)

  
   
    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, heads = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=device))[None, :, :].expand(b, t, heads)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)

