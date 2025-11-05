"""
AD4CD (Anomaly Detection for Cognitive Diagnosis) Model

Integrates anomaly detection with cognitive diagnosis to identify:
- Abnormal learning patterns
- Cheating behavior
- Guessing patterns
- Careless mistakes
- Unusual performance fluctuations

Reference: https://github.com/BIMK/Intelligent-Education/tree/main/AD4CD
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available for AD4CD. Using mock implementation.")


class AD4CD_Model(nn.Module if TORCH_AVAILABLE else object):
    """
    AD4CD Model: Anomaly Detection for Cognitive Diagnosis
    
    Combines cognitive diagnosis with anomaly detection to:
    1. Detect abnormal response patterns
    2. Identify potential cheating or guessing
    3. Improve diagnosis accuracy by filtering anomalies
    """
    
    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        embedding_dim: int = 64,
        anomaly_threshold: float = 0.7,
        device: str = 'cpu'
    ):
        """
        Initialize AD4CD model
        
        Args:
            num_students: Number of students
            num_exercises: Number of exercises/questions
            num_concepts: Number of knowledge concepts
            embedding_dim: Dimension of embeddings
            anomaly_threshold: Threshold for anomaly detection (0-1)
            device: 'cpu' or 'cuda'
        """
        if TORCH_AVAILABLE:
            super(AD4CD_Model, self).__init__()
        
        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.anomaly_threshold = anomaly_threshold
        self.device = device if TORCH_AVAILABLE else 'cpu'
        
        if TORCH_AVAILABLE:
            # Student embeddings
            self.student_emb = nn.Embedding(num_students + 1, embedding_dim, padding_idx=0)
            
            # Exercise embeddings
            self.exercise_emb = nn.Embedding(num_exercises + 1, embedding_dim, padding_idx=0)
            
            # Concept embeddings
            self.concept_emb = nn.Embedding(num_concepts + 1, embedding_dim, padding_idx=0)
            
            # Cognitive diagnosis network
            self.cd_network = nn.Sequential(
                nn.Linear(embedding_dim * 3, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            # Anomaly detection network (ECOD-inspired)
            self.ad_network = nn.Sequential(
                nn.Linear(embedding_dim * 3, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            self.to(self.device)
    
    def forward(
        self,
        student_ids: 'torch.Tensor',
        exercise_ids: 'torch.Tensor',
        concept_ids: 'torch.Tensor'
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Forward pass
        
        Args:
            student_ids: Student IDs (batch_size,)
            exercise_ids: Exercise IDs (batch_size,)
            concept_ids: Concept IDs (batch_size,)
            
        Returns:
            performance_pred: Predicted performance (batch_size, 1)
            anomaly_score: Anomaly score (batch_size, 1)
        """
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch required for AD4CD")
        
        # Get embeddings
        student_emb = self.student_emb(student_ids)
        exercise_emb = self.exercise_emb(exercise_ids)
        concept_emb = self.concept_emb(concept_ids)
        
        # Concatenate embeddings
        combined = torch.cat([student_emb, exercise_emb, concept_emb], dim=-1)
        
        # Cognitive diagnosis prediction
        performance_pred = self.cd_network(combined)
        
        # Anomaly detection
        anomaly_score = self.ad_network(combined)
        
        return performance_pred, anomaly_score
    
    def detect_anomalies(
        self,
        student_ids: 'torch.Tensor',
        exercise_ids: 'torch.Tensor',
        concept_ids: 'torch.Tensor',
        responses: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Detect anomalous responses
        
        Args:
            student_ids: Student IDs
            exercise_ids: Exercise IDs
            concept_ids: Concept IDs
            responses: Actual responses (0/1)
            
        Returns:
            is_anomaly: Boolean tensor indicating anomalies
        """
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch required for AD4CD")
        
        with torch.no_grad():
            performance_pred, anomaly_score = self.forward(
                student_ids, exercise_ids, concept_ids
            )
            
            # Anomaly if:
            # 1. High anomaly score from AD network
            # 2. Large discrepancy between prediction and actual response
            discrepancy = torch.abs(performance_pred.squeeze() - responses.float())
            
            is_anomaly = (anomaly_score.squeeze() > self.anomaly_threshold) | \
                        (discrepancy > 0.5)
            
            return is_anomaly


class AD4CD_CognitiveDiagnosis:
    """
    AD4CD Cognitive Diagnosis Service
    
    Integrates anomaly detection with cognitive diagnosis for improved accuracy
    """
    
    def __init__(
        self,
        num_students: int = 1000,
        num_exercises: int = 1000,
        num_concepts: int = 500,
        embedding_dim: int = 64,
        anomaly_threshold: float = 0.7,
        device: str = 'cpu'
    ):
        """Initialize AD4CD cognitive diagnosis"""
        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.anomaly_threshold = anomaly_threshold
        self.device = device
        
        if TORCH_AVAILABLE:
            self.model = AD4CD_Model(
                num_students=num_students,
                num_exercises=num_exercises,
                num_concepts=num_concepts,
                embedding_dim=embedding_dim,
                anomaly_threshold=anomaly_threshold,
                device=device
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()
        else:
            self.model = None
            print("⚠️  AD4CD running in mock mode (PyTorch not available)")
    
    def diagnose_with_anomaly_detection(
        self,
        student_id: int,
        exercise_id: int,
        concept_id: int,
        response: int
    ) -> Dict:
        """
        Diagnose student performance with anomaly detection
        
        Args:
            student_id: Student ID
            exercise_id: Exercise ID
            concept_id: Concept ID
            response: Student response (0/1)
            
        Returns:
            Dictionary with diagnosis results and anomaly info
        """
        if not TORCH_AVAILABLE:
            # Mock implementation
            return {
                'predicted_performance': 0.5 + np.random.randn() * 0.1,
                'anomaly_score': np.random.rand(),
                'is_anomaly': False,
                'confidence': 0.8,
                'diagnosis': 'normal'
            }
        
        self.model.eval()
        with torch.no_grad():
            student_tensor = torch.tensor([student_id], device=self.device)
            exercise_tensor = torch.tensor([exercise_id], device=self.device)
            concept_tensor = torch.tensor([concept_id], device=self.device)
            response_tensor = torch.tensor([response], dtype=torch.float32, device=self.device)
            
            # Get predictions
            performance_pred, anomaly_score = self.model(
                student_tensor, exercise_tensor, concept_tensor
            )
            
            # Detect anomaly
            is_anomaly = self.model.detect_anomalies(
                student_tensor, exercise_tensor, concept_tensor, response_tensor
            )
            
            # Calculate confidence (lower if anomaly detected)
            confidence = 1.0 - anomaly_score.item() if is_anomaly.item() else 0.9
            
            # Determine diagnosis
            if is_anomaly.item():
                if response == 1 and performance_pred.item() < 0.3:
                    diagnosis = 'possible_cheating'
                elif response == 0 and performance_pred.item() > 0.7:
                    diagnosis = 'careless_mistake'
                else:
                    diagnosis = 'guessing'
            else:
                diagnosis = 'normal'
            
            return {
                'predicted_performance': performance_pred.item(),
                'anomaly_score': anomaly_score.item(),
                'is_anomaly': is_anomaly.item(),
                'confidence': confidence,
                'diagnosis': diagnosis
            }
    
    def train_step(
        self,
        student_ids: List[int],
        exercise_ids: List[int],
        concept_ids: List[int],
        responses: List[int]
    ) -> float:
        """
        Train the model for one step
        
        Returns:
            Training loss
        """
        if not TORCH_AVAILABLE:
            return 0.5  # Mock loss
        
        self.model.train()
        
        # Convert to tensors
        student_tensor = torch.tensor(student_ids, device=self.device)
        exercise_tensor = torch.tensor(exercise_ids, device=self.device)
        concept_tensor = torch.tensor(concept_ids, device=self.device)
        response_tensor = torch.tensor(responses, dtype=torch.float32, device=self.device)
        
        # Forward pass
        performance_pred, anomaly_score = self.model(
            student_tensor, exercise_tensor, concept_tensor
        )
        
        # Calculate loss
        cd_loss = self.criterion(performance_pred.squeeze(), response_tensor)
        
        # Anomaly detection loss (unsupervised)
        # Encourage low anomaly scores for normal patterns
        ad_loss = anomaly_score.mean()
        
        # Combined loss
        total_loss = cd_loss + 0.1 * ad_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def save_model(self, path: str):
        """Save model to file"""
        if TORCH_AVAILABLE and self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': {
                    'num_students': self.num_students,
                    'num_exercises': self.num_exercises,
                    'num_concepts': self.num_concepts,
                    'embedding_dim': self.embedding_dim,
                    'anomaly_threshold': self.anomaly_threshold
                }
            }, path)
            print(f"✅ AD4CD model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        if TORCH_AVAILABLE and self.model:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ AD4CD model loaded from {path}")


# Backward compatibility
NCD_Model = AD4CD_Model
NCD_CognitiveDiagnosis = AD4CD_CognitiveDiagnosis

