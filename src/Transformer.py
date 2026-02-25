import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int
    d_vocab: int
    d_hidden: int
    n_context: int
    n_layers: int

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # self.W_qk = nn.Linear(config.d_model, config.d_vocab)
        self.bilinear = nn.Bilinear(config.d_model, config.d_model, config.n_context, bias=False)
        self.M = torch.triu(torch.ones((config.n_context, config.n_context)), diagonal=1)
        self.M = self.M.masked_fill(self.M.bool(), -torch.inf)
        self.second_matmult = nn.Linear(config.d_model, config.d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        xwx = self.bilinear(x, x) # d_m x d_m
        x_masked = xwx+ self.M 
        x_softmaxed = self.softmax(x_masked)
        x_fin = x_softmaxed@x
        #multiply softmaxed by x
        #multiply that by wov
        x_fin = self.second_matmult(x_fin)
        return x_fin

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear_up = nn.Linear(config.d_model, config.d_hidden)
        self.linear_down = nn.Linear(config.d_hidden, config.d_model)
    
    def forward(self, x):
        x = self.linear_up(x)
        x = torch.relu(x)
        x = self.linear_down(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.MLP = MLP(config=self.config)
        self.Attention = Attention(config=self.config)
    
    def forward(self, x):
        return x + self.Attention(x) + self.MLP(x)
    
class Transformer(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.d_vocab, embedding_dim=config.d_model)
        self.transformerBlock = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        for i, l in enumerate(self.transformerBlock):
            x = self.transformerBlock[i](x)
        return x