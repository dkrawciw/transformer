import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass

#config for the transformer
class Config:
    d_model: int
    d_vocab: int
    d_hidden: int
    d_head: int
    n_context: int
    n_layers: int

#attention head
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.Wk = nn.Linear(config.d_model, config.d_head, bias=False)
        self.Wq = nn.Linear(config.d_model, config.d_head, bias=False)
        self.M = torch.triu(torch.ones((config.n_context, config.n_context)), diagonal=1)
        self.M = self.M.masked_fill(self.M.bool(), -torch.inf)
        self.second_matmult = nn.Linear(config.d_model, config.d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        xwk = self.Wk(x)
        xwq = self.Wq(x)
        xwx = xwq @ xwk.T
        x_masked = xwx+ self.M 
        x_softmaxed = self.softmax(x_masked)
        x_fin = x_softmaxed@x
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

#transformer block x + A(x) + MLP(X)    
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
        self.config = config
        #embedding and positional embedding
        self.embedding = nn.Embedding(num_embeddings=config.d_vocab, embedding_dim=config.d_model)
        self.pos_embedding = nn.Embedding(config.n_context, config.d_model)
        self.transformerBlock = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding(torch.arange(x.shape[0]))
  
        for i, l in enumerate(self.transformerBlock):
            x = self.transformerBlock[i](x)

        #unembedding step
        x = x @ self.embedding.weight.T
        return x