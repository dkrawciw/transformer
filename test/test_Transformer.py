import pytest
import torch
import torch.nn as nn

from src.Transformer import Config, Attention, MLP, TransformerBlock, Transformer

config = Config(
    d_model = 5,
    d_vocab = 10,
    d_hidden = 15,
    n_context = 15,
    n_layers = 2,
)

class TestAttention:
    def test_attention_size(self):
        x0 = torch.randn(config.n_context, config.d_model)
        attention = Attention(config)
        x1 = attention(x0)
        assert x1.shape == (config.n_context, config.d_model)

class TestMLP:
    def test_MLP_size(self):
        x0 = torch.randn((1, config.n_context, config.d_model))
        mlp = MLP(config)
        x1 = mlp(x0)
        assert x1.shape == (1, config.n_context, config.d_model)

class TestTransformerBlock:
    def test_transformerblock_size(self):
        x0 = torch.randn(config.n_context, config.d_model)
        block = TransformerBlock(config)
        x1 = block(x0)
        assert x1.shape == (config.n_context, config.d_model)

class TestTransformer:
    def test_transformer_size(self):
        x0 = torch.randint(low=0, high=config.d_vocab, size = (1, 4))
        tf = Transformer(config=config)
        x1 = tf(x0)
        assert x1.shape == (config.n_context, config.d_model)