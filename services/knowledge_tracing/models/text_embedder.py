# services/knowledge_tracing/models/text_embedder.py

import torch
import torch.nn as nn

class TextEmbedder(nn.Module):
    """Simulates embedding natural language text (like a student question)."""
    def __init__(self, embedding_dim):
        super(TextEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, text: str):
        if not text: 
            return torch.zeros(1, self.embedding_dim)
        # Simulate embedding by creating a deterministic "random" vector
        torch.manual_seed(len(text))
        return torch.randn(1, self.embedding_dim)
