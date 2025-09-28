# services/knowledge_tracing/models/astnn_model.py

import torch
import torch.nn as nn
import ast

class ASTNN(nn.Module):
    """Simulates converting an AST into a vector embedding (Knowledge Representation)."""
    def __init__(self, embedding_dim):
        super(ASTNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(10, embedding_dim)

    def forward(self, code_submission: str):
        if not code_submission: 
            return torch.zeros(1, self.embedding_dim)
        try:
            tree = ast.parse(code_submission)
            num_nodes = len(list(ast.walk(tree)))
            num_func = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
            features = torch.tensor([len(code_submission), num_nodes, num_func, 0,0,0,0,0,0,0], dtype=torch.float32)
            return self.projection(features).unsqueeze(0)
        except (SyntaxError, ValueError):
            return torch.zeros(1, self.embedding_dim)
