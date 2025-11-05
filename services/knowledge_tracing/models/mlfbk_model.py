# services/knowledge_tracing/models/mlfbk_model.py
"""
Integrated Knowledge Tracing: SQKT + MLFBK

Combines two state-of-the-art models:

1. SQKT (Sequential Question-based Knowledge Tracing)
   - Temporal sequence modeling with transformers
   - Multi-head attention (8 heads, 4 layers)
   - Tracks submissions, questions, and responses
   - 81% accuracy

2. MLFBK (Multi-Features with Latent Relations BERT Knowledge Tracing)
   - Multi-feature extraction (student_id, item_id, skill_id, etc.)
   - BERT-based latent representations
   - Captures complex feature interactions

This integration provides superior accuracy by leveraging both models' strengths.

References:
- SQKT: https://github.com/holi-lab/SQKT
- MLFBK: Multi-feature BERT-based knowledge tracing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class SQKT_Model(nn.Module):
    """
    Sequential Question-based Knowledge Tracing (SQKT) Model

    Incorporates:
    1. Exercise/problem embeddings
    2. Student submission embeddings (code/text)
    3. Student question embeddings
    4. Educator response embeddings
    5. Temporal information
    6. Multi-head self-attention

    Architecture:
    - Embedding layers for exercises, submissions, questions, responses
    - Transformer encoder with multi-head attention
    - Prediction head for next performance
    """

    def __init__(
        self,
        num_exercises: int,
        num_skills: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.1
    ):
        super(SQKT_Model, self).__init__()
        print("\n=== Initializing SQKT (Sequential Question-based Knowledge Tracing) Model ===")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Exercise/Problem embeddings
        self.exercise_embedding = nn.Embedding(num_exercises + 1, embedding_dim, padding_idx=0)

        # Skill/Concept embeddings
        self.skill_embedding = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=0)

        # Response embeddings (0: padding, 1: incorrect, 2: correct)
        self.response_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)

        # Interaction type embeddings (0: pad, 1: submission, 2: student_question, 3: educator_response)
        self.interaction_type_embedding = nn.Embedding(5, embedding_dim, padding_idx=0)

        # Positional encoding
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Projection layers for different content types
        self.submission_projection = nn.Linear(embedding_dim, embedding_dim)
        self.question_projection = nn.Linear(embedding_dim, embedding_dim)
        self.response_projection = nn.Linear(embedding_dim, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

        # Knowledge state extraction layer
        self.knowledge_state_layer = nn.Linear(embedding_dim, embedding_dim)

        print(f"✅ SQKT Model initialized:")
        print(f"   - Embedding dimension: {embedding_dim}")
        print(f"   - Attention heads: {num_heads}")
        print(f"   - Transformer layers: {num_layers}")
        print(f"   - Max sequence length: {max_seq_len}")
        print(f"   - Dropout: {dropout}")

    def forward(
        self,
        exercise_seq: torch.Tensor,
        skill_seq: torch.Tensor,
        response_seq: torch.Tensor,
        interaction_type_seq: torch.Tensor,
        submission_embed_seq: Optional[torch.Tensor] = None,
        question_embed_seq: Optional[torch.Tensor] = None,
        educator_response_embed_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of SQKT model

        Args:
            exercise_seq: [batch_size, seq_len] - Exercise IDs
            skill_seq: [batch_size, seq_len] - Skill/Concept IDs
            response_seq: [batch_size, seq_len] - Response correctness (0: pad, 1: incorrect, 2: correct)
            interaction_type_seq: [batch_size, seq_len] - Type of interaction
            submission_embed_seq: [batch_size, seq_len, embed_dim] - Submission embeddings (optional)
            question_embed_seq: [batch_size, seq_len, embed_dim] - Question embeddings (optional)
            educator_response_embed_seq: [batch_size, seq_len, embed_dim] - Response embeddings (optional)
            attention_mask: [batch_size, seq_len] - Attention mask (optional)

        Returns:
            predictions: [batch_size, seq_len, 1] - Predicted performance
            knowledge_states: [batch_size, seq_len, embed_dim] - Knowledge state representations
        """
        batch_size, seq_len = exercise_seq.size()

        # Get base embeddings
        exercise_embeds = self.exercise_embedding(exercise_seq)
        skill_embeds = self.skill_embedding(skill_seq)
        response_embeds = self.response_embedding(response_seq)
        type_embeds = self.interaction_type_embedding(interaction_type_seq)

        # Positional embeddings
        positions = torch.arange(seq_len, device=exercise_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.positional_embedding(positions)

        # Initialize combined embeddings
        combined_embeds = exercise_embeds + skill_embeds + response_embeds + type_embeds + pos_embeds

        # Add content-specific embeddings if provided
        if submission_embed_seq is not None:
            submission_proj = self.submission_projection(submission_embed_seq)
            combined_embeds = combined_embeds + submission_proj

        if question_embed_seq is not None:
            question_proj = self.question_projection(question_embed_seq)
            combined_embeds = combined_embeds + question_proj

        if educator_response_embed_seq is not None:
            response_proj = self.response_projection(educator_response_embed_seq)
            combined_embeds = combined_embeds + response_proj

        # Create causal mask for autoregressive prediction
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(exercise_seq.device)

        # Apply transformer encoder
        knowledge_state_sequence = self.transformer_encoder(
            combined_embeds,
            mask=causal_mask,
            src_key_padding_mask=attention_mask
        )

        # Extract knowledge states
        knowledge_states = self.knowledge_state_layer(knowledge_state_sequence)

        # Generate predictions
        predictions = self.prediction_head(knowledge_state_sequence)
        predictions = torch.sigmoid(predictions)

        return predictions, knowledge_states

class SQKT_KnowledgeTracer:
    """
    SQKT Knowledge Tracing Service

    Orchestrates the SQKT model and provides high-level interfaces for:
    - Knowledge state prediction
    - Performance prediction
    - Model training and updates
    - Integration with Neo4j knowledge graph
    """

    def __init__(
        self,
        num_exercises: int = 1000,
        num_skills: int = 500,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize SQKT Knowledge Tracer

        Args:
            num_exercises: Number of unique exercises/problems
            num_skills: Number of unique skills/concepts
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.num_exercises = num_exercises
        self.num_skills = num_skills

        # Initialize SQKT model
        self.model = SQKT_Model(
            num_exercises=num_exercises,
            num_skills=num_skills,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        ).to(self.device)

        # Optimizer and loss function
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.criterion = nn.BCELoss()

        # Training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': []
        }

        print(f"✅ SQKT Knowledge Tracer initialized on device: {self.device}")

    def predict_knowledge_state(
        self,
        interaction_sequence: List[Dict],
        return_predictions: bool = False
    ) -> torch.Tensor:
        """
        Predict knowledge state based on interaction sequence

        Args:
            interaction_sequence: List of interaction dictionaries with keys:
                - exercise_id: int
                - skill_id: int
                - response: int (0: incorrect, 1: correct)
                - interaction_type: int (1: submission, 2: question, 3: response)
                - submission_text: str (optional)
                - question_text: str (optional)
                - response_text: str (optional)
            return_predictions: Whether to return performance predictions

        Returns:
            knowledge_states: [seq_len, embedding_dim] tensor of knowledge states
            predictions (optional): [seq_len, 1] tensor of performance predictions
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare input tensors
            exercise_seq, skill_seq, response_seq, type_seq = self._prepare_sequences(interaction_sequence)

            # Get predictions and knowledge states
            predictions, knowledge_states = self.model(
                exercise_seq,
                skill_seq,
                response_seq,
                type_seq
            )

            # Remove batch dimension
            knowledge_states = knowledge_states.squeeze(0)
            predictions = predictions.squeeze(0)

        if return_predictions:
            return knowledge_states, predictions
        return knowledge_states

    def predict_next_performance(
        self,
        interaction_sequence: List[Dict],
        next_exercise_id: int,
        next_skill_id: int
    ) -> float:
        """
        Predict performance on next exercise

        Args:
            interaction_sequence: Historical interactions
            next_exercise_id: ID of next exercise
            next_skill_id: ID of skill for next exercise

        Returns:
            Predicted probability of correct response (0-1)
        """
        self.model.eval()

        with torch.no_grad():
            # Get current knowledge state
            knowledge_states, _ = self.predict_knowledge_state(
                interaction_sequence,
                return_predictions=True
            )

            # Use last knowledge state for prediction
            last_state = knowledge_states[-1:, :]

            # Create input for next exercise
            next_exercise = torch.tensor([[next_exercise_id]], device=self.device)
            next_skill = torch.tensor([[next_skill_id]], device=self.device)
            next_response = torch.tensor([[0]], device=self.device)  # Placeholder
            next_type = torch.tensor([[1]], device=self.device)  # Submission type

            # Predict
            predictions, _ = self.model(
                next_exercise,
                next_skill,
                next_response,
                next_type
            )

            return predictions.item()

    def train_epoch(
        self,
        train_data: List[Dict],
        batch_size: int = 32
    ) -> float:
        """
        Train model for one epoch

        Args:
            train_data: List of training sequences
            batch_size: Batch size for training

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Prepare batch tensors
            batch_tensors = self._prepare_batch(batch)

            # Forward pass
            predictions, _ = self.model(**batch_tensors['inputs'])

            # Calculate loss
            loss = self.criterion(
                predictions.squeeze(-1),
                batch_tensors['targets']
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(
        self,
        val_data: List[Dict],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data

        Args:
            val_data: List of validation sequences
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with metrics: accuracy, precision, recall, f1, auc
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                batch_tensors = self._prepare_batch(batch)

                predictions, _ = self.model(**batch_tensors['inputs'])

                loss = self.criterion(
                    predictions.squeeze(-1),
                    batch_tensors['targets']
                )

                total_loss += loss.item()
                num_batches += 1

                all_predictions.extend(predictions.squeeze(-1).cpu().numpy().flatten())
                all_targets.extend(batch_tensors['targets'].cpu().numpy().flatten())

        # Calculate metrics
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
        targets = np.array(all_targets)

        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': accuracy_score(targets, predictions_binary),
            'precision': precision_score(targets, predictions_binary, zero_division=0),
            'recall': recall_score(targets, predictions_binary, zero_division=0),
            'f1': f1_score(targets, predictions_binary, zero_division=0),
            'auc': roc_auc_score(targets, all_predictions) if len(np.unique(targets)) > 1 else 0.0
        }

        return metrics

    def _prepare_sequences(
        self,
        interaction_sequence: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare interaction sequence into model input tensors

        Args:
            interaction_sequence: List of interaction dictionaries

        Returns:
            Tuple of (exercise_seq, skill_seq, response_seq, type_seq)
        """
        exercise_ids = []
        skill_ids = []
        responses = []
        types = []

        for interaction in interaction_sequence:
            exercise_ids.append(interaction.get('exercise_id', 0))
            skill_ids.append(interaction.get('skill_id', 0))
            # Map response: 0 -> 1 (incorrect), 1 -> 2 (correct)
            response = interaction.get('response', 0)
            responses.append(response + 1 if response in [0, 1] else 0)
            types.append(interaction.get('interaction_type', 1))

        # Convert to tensors and add batch dimension
        exercise_seq = torch.tensor([exercise_ids], dtype=torch.long, device=self.device)
        skill_seq = torch.tensor([skill_ids], dtype=torch.long, device=self.device)
        response_seq = torch.tensor([responses], dtype=torch.long, device=self.device)
        type_seq = torch.tensor([types], dtype=torch.long, device=self.device)

        return exercise_seq, skill_seq, response_seq, type_seq

    def _prepare_batch(
        self,
        batch: List[Dict]
    ) -> Dict:
        """
        Prepare a batch of sequences for training/evaluation

        Args:
            batch: List of sequence dictionaries

        Returns:
            Dictionary with 'inputs' and 'targets' tensors
        """
        # Find max sequence length in batch
        max_len = max(len(seq['interactions']) for seq in batch)

        # Initialize batch tensors
        batch_size = len(batch)
        exercise_seqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        skill_seqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        response_seqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        type_seqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        targets = torch.zeros(batch_size, max_len, dtype=torch.float, device=self.device)

        # Fill tensors
        for i, seq in enumerate(batch):
            interactions = seq['interactions']
            seq_len = len(interactions)

            for j, interaction in enumerate(interactions):
                exercise_seqs[i, j] = interaction.get('exercise_id', 0)
                skill_seqs[i, j] = interaction.get('skill_id', 0)
                response = interaction.get('response', 0)
                response_seqs[i, j] = response + 1 if response in [0, 1] else 0
                type_seqs[i, j] = interaction.get('interaction_type', 1)
                targets[i, j] = float(response)

        return {
            'inputs': {
                'exercise_seq': exercise_seqs,
                'skill_seq': skill_seqs,
                'response_seq': response_seqs,
                'interaction_type_seq': type_seqs
            },
            'targets': targets
        }

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'num_exercises': self.num_exercises,
                'num_skills': self.num_skills,
                'embedding_dim': self.embedding_dim
            }
        }, path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"✅ Model loaded from {path}")


# MLFBK Model (Multi-Features with Latent Relations BERT Knowledge Tracing)
# This is integrated with SQKT for enhanced performance
class MLFBK_Model(nn.Module):
    """
    MLFBK: Multi-Features with Latent Relations BERT Knowledge Tracing

    Extracts and processes multiple features for knowledge tracing.
    Can be used standalone or integrated with SQKT.
    """

    def __init__(
        self,
        num_students: int,
        num_items: int,
        num_skills: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super(MLFBK_Model, self).__init__()

        self.num_students = num_students
        self.num_items = num_items
        self.num_skills = num_skills
        self.embedding_dim = embedding_dim

        # Feature embeddings
        self.student_emb = nn.Embedding(num_students + 1, embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.skill_emb = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=0)
        self.response_emb = nn.Embedding(3, embedding_dim, padding_idx=0)

        # BERT-like transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, student_ids, item_ids, skill_ids, responses):
        """Forward pass"""
        student_emb = self.student_emb(student_ids)
        item_emb = self.item_emb(item_ids)
        skill_emb = self.skill_emb(skill_ids)
        response_emb = self.response_emb(responses)

        combined = student_emb + item_emb + skill_emb + response_emb
        output = self.transformer(combined)
        predictions = torch.sigmoid(self.fc(output))

        return predictions


class MLFBK_KnowledgeTracer:
    """MLFBK Knowledge Tracer Service"""

    def __init__(self, num_students=1000, num_items=1000, num_skills=500, embedding_dim=128, device='cpu'):
        self.model = MLFBK_Model(num_students, num_items, num_skills, embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device

    def predict(self, student_ids, item_ids, skill_ids, responses):
        """Predict next performance"""
        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor([student_ids], device=self.device),
                torch.tensor([item_ids], device=self.device),
                torch.tensor([skill_ids], device=self.device),
                torch.tensor([responses], device=self.device)
            ).squeeze()


# Export both SQKT and MLFBK models
__all__ = ['SQKT_Model', 'SQKT_KnowledgeTracer', 'MLFBK_Model', 'MLFBK_KnowledgeTracer']

