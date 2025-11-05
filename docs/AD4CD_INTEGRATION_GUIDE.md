# AD4CD Integration Guide

## Overview

AD4CD (Anomaly Detection for Cognitive Diagnosis) has been integrated into KTCD_Aug to enhance cognitive diagnosis accuracy by detecting and filtering anomalous student responses.

**Reference**: https://github.com/BIMK/Intelligent-Education/tree/main/AD4CD

---

## What is AD4CD?

AD4CD combines **Cognitive Diagnosis** with **Anomaly Detection** to:

1. **Detect Abnormal Patterns**
   - Cheating behavior (correct answers with low predicted performance)
   - Guessing patterns (inconsistent performance)
   - Careless mistakes (incorrect answers with high predicted performance)
   - Unusual performance fluctuations

2. **Improve Diagnosis Accuracy**
   - Filter out anomalous responses
   - Confidence-weighted mastery updates
   - More reliable knowledge state estimation

3. **Provide Actionable Insights**
   - Identify students who may need intervention
   - Detect potential academic integrity issues
   - Understand learning patterns better

---

## Architecture

### AD4CD Model Components

```
┌─────────────────────────────────────────────────────────┐
│                    AD4CD Architecture                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Embeddings (3 types):                            │
│  ├─ Student Embeddings (num_students × 64)              │
│  ├─ Exercise Embeddings (num_exercises × 64)            │
│  └─ Concept Embeddings (num_concepts × 64)              │
│                                                          │
│  Cognitive Diagnosis Network:                           │
│  ├─ Linear(192 → 128) + ReLU + Dropout(0.2)            │
│  ├─ Linear(128 → 64) + ReLU + Dropout(0.2)             │
│  └─ Linear(64 → 1) + Sigmoid                           │
│  Output: Performance Prediction (0-1)                   │
│                                                          │
│  Anomaly Detection Network (ECOD-inspired):             │
│  ├─ Linear(192 → 128) + ReLU                           │
│  ├─ Linear(128 → 64) + ReLU                            │
│  └─ Linear(64 → 1) + Sigmoid                           │
│  Output: Anomaly Score (0-1)                            │
│                                                          │
│  Anomaly Detection Logic:                               │
│  ├─ High anomaly score (> threshold)                    │
│  ├─ Large prediction-response discrepancy (> 0.5)       │
│  └─ Classification: normal/cheating/guessing/careless   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Integration with KTCD_Aug

```
┌─────────────────────────────────────────────────────────┐
│              AD4CD Integration Flow                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Student Response                                        │
│  └─> AD4CD Diagnosis                                    │
│       ├─ Performance Prediction                         │
│       ├─ Anomaly Score                                  │
│       ├─ Anomaly Classification                         │
│       └─ Confidence Score                               │
│                                                          │
│  If NOT Anomalous:                                       │
│  ├─> Update SQKT (Knowledge Tracing)                    │
│  ├─> Update Neo4j (Mastery with confidence)             │
│  └─> Store Diagnosis                                    │
│                                                          │
│  If Anomalous:                                           │
│  ├─> Flag for review                                    │
│  ├─> Store with anomaly label                           │
│  └─> Skip mastery update (or reduce weight)             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Usage

### 1. Initialize AD4CD

```python
from services.cognitive_diagnosis import AD4CD_CognitiveDiagnosis, AD4CDIntegrationService

# Initialize AD4CD model
ad4cd_diagnosis = AD4CD_CognitiveDiagnosis(
    num_students=1000,
    num_exercises=1000,
    num_concepts=500,
    embedding_dim=64,
    anomaly_threshold=0.7,  # Threshold for anomaly detection
    device='cpu'  # or 'cuda' for GPU
)

# Initialize integration service
ad4cd_service = AD4CDIntegrationService(
    ad4cd_diagnosis=ad4cd_diagnosis,
    neo4j_service=your_neo4j_service,
    sqkt_service=your_sqkt_service
)
```

### 2. Diagnose Student Responses

```python
# Student submits quiz answer
diagnosis = ad4cd_service.diagnose_response(
    student_id='roma',
    exercise_name='MongoDB Query Quiz',
    concept_name='NoSQL Queries',
    response=1  # 1=correct, 0=incorrect
)

print(f"Predicted Performance: {diagnosis['predicted_performance']:.2%}")
print(f"Anomaly Score: {diagnosis['anomaly_score']:.2%}")
print(f"Is Anomaly: {diagnosis['is_anomaly']}")
print(f"Diagnosis: {diagnosis['diagnosis']}")
print(f"Confidence: {diagnosis['confidence']:.2%}")
```

**Output Example**:
```
Predicted Performance: 78%
Anomaly Score: 15%
Is Anomaly: False
Diagnosis: normal
Confidence: 85%
```

### 3. Get Anomaly Report

```python
# Get student's anomaly statistics
report = ad4cd_service.get_student_anomaly_report('roma')

print(f"Total Responses: {report['total_responses']}")
print(f"Anomalous Responses: {report['anomalous_responses']}")
print(f"Anomaly Rate: {report['anomaly_rate']:.2%}")
print(f"Cheating Incidents: {report['cheating_incidents']}")
print(f"Guessing Incidents: {report['guessing_incidents']}")
print(f"Careless Mistakes: {report['careless_mistakes']}")
```

**Output Example**:
```
Total Responses: 50
Anomalous Responses: 3
Anomaly Rate: 6%
Cheating Incidents: 0
Guessing Incidents: 2
Careless Mistakes: 1
```

### 4. Get Filtered Performance Data

```python
# Get performance data without anomalies
clean_data = ad4cd_service.get_filtered_performance_data(
    student_id='roma',
    filter_anomalies=True  # Filter out anomalous responses
)

for record in clean_data:
    print(f"{record['exercise']}: {record['response']} (predicted: {record['predicted']:.2f})")
```

---

## Anomaly Types

### 1. Normal Response
- **Criteria**: Low anomaly score, prediction matches response
- **Action**: Update mastery normally
- **Confidence**: High (0.9)

### 2. Possible Cheating
- **Criteria**: Correct response (1) but low predicted performance (<0.3)
- **Action**: Flag for review, skip mastery update
- **Confidence**: Low (based on anomaly score)
- **Example**: Student gets difficult question correct despite low mastery

### 3. Guessing
- **Criteria**: High anomaly score, inconsistent with history
- **Action**: Reduce confidence in mastery update
- **Confidence**: Medium
- **Example**: Random correct/incorrect pattern

### 4. Careless Mistake
- **Criteria**: Incorrect response (0) but high predicted performance (>0.7)
- **Action**: Partial mastery update with reduced weight
- **Confidence**: Medium
- **Example**: Student makes simple error despite high mastery

---

## Training AD4CD

### Prepare Training Data

```python
# Format: Lists of student responses
student_ids = [1, 1, 1, 2, 2, 2, ...]
exercise_ids = [10, 11, 12, 10, 11, 12, ...]
concept_ids = [5, 5, 6, 5, 5, 6, ...]
responses = [1, 0, 1, 1, 1, 0, ...]  # 1=correct, 0=incorrect
```

### Train Model

```python
# Training loop
num_epochs = 20
batch_size = 32

for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(len(student_ids))
    
    # Train in batches
    for i in range(0, len(student_ids), batch_size):
        batch_indices = indices[i:i+batch_size]
        
        loss = ad4cd_diagnosis.train_step(
            student_ids=[student_ids[j] for j in batch_indices],
            exercise_ids=[exercise_ids[j] for j in batch_indices],
            concept_ids=[concept_ids[j] for j in batch_indices],
            responses=[responses[j] for j in batch_indices]
        )
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Save trained model
ad4cd_diagnosis.save_model('models/ad4cd_checkpoint.pt')
```

### Load Trained Model

```python
ad4cd_diagnosis.load_model('models/ad4cd_checkpoint.pt')
```

---

## Integration with SQKT

AD4CD works seamlessly with SQKT for enhanced knowledge tracing:

```python
# Initialize both systems
sqkt_service = SQKTIntegrationService(sqkt_tracer, neo4j_service)
ad4cd_service = AD4CDIntegrationService(ad4cd_diagnosis, neo4j_service, sqkt_service)

# Process student response
diagnosis = ad4cd_service.diagnose_response(
    student_id='roma',
    exercise_name='Quiz 1',
    concept_name='Concept A',
    response=1
)

# If not anomalous, SQKT is automatically updated
# If anomalous, SQKT update is skipped
```

---

## Neo4j Schema

### Diagnosis Node

```cypher
CREATE (d:Diagnosis {
    exercise_name: 'MongoDB Query Quiz',
    concept_name: 'NoSQL Queries',
    response: 1,
    predicted_performance: 0.78,
    anomaly_score: 0.15,
    is_anomaly: false,
    confidence: 0.85,
    diagnosis: 'normal',
    timestamp: '2025-01-15T10:30:00'
})
```

### Relationships

```cypher
// Student has diagnosis
(Student)-[:HAS_DIAGNOSIS]->(Diagnosis)

// Confidence-weighted mastery
(Student)-[k:KNOWS {
    mastery_level: 0.78,
    confidence: 0.85,
    last_updated: '2025-01-15T10:30:00'
}]->(Concept)
```

---

## Best Practices

### 1. Anomaly Threshold Selection
- **Conservative (0.8)**: Fewer false positives, may miss some anomalies
- **Moderate (0.7)**: Balanced detection (recommended)
- **Aggressive (0.6)**: More sensitive, may flag normal variations

### 2. Confidence Weighting
- Use confidence scores to weight mastery updates
- Lower confidence for anomalous responses
- Higher confidence for consistent patterns

### 3. Regular Retraining
- Retrain AD4CD periodically with new data
- Adapt to changing student behaviors
- Improve detection accuracy over time

### 4. Human Review
- Flag high-anomaly responses for manual review
- Use AD4CD as a tool, not absolute truth
- Combine with other integrity measures

---

## API Reference

### AD4CD_CognitiveDiagnosis

```python
class AD4CD_CognitiveDiagnosis:
    def __init__(num_students, num_exercises, num_concepts, embedding_dim=64, 
                 anomaly_threshold=0.7, device='cpu')
    def diagnose_with_anomaly_detection(student_id, exercise_id, concept_id, response)
    def train_step(student_ids, exercise_ids, concept_ids, responses)
    def save_model(path)
    def load_model(path)
```

### AD4CDIntegrationService

```python
class AD4CDIntegrationService:
    def __init__(ad4cd_diagnosis, neo4j_service=None, sqkt_service=None)
    def diagnose_response(student_id, exercise_name, concept_name, response, timestamp=None)
    def get_student_anomaly_report(student_id)
    def update_mastery_with_confidence(student_id, concept_name, performance, confidence)
    def get_filtered_performance_data(student_id, filter_anomalies=True)
```

---

## Troubleshooting

### Issue: High False Positive Rate

**Solutions:**
- Increase anomaly_threshold (e.g., 0.7 → 0.8)
- Collect more training data
- Retrain model with balanced dataset

### Issue: Missing Anomalies

**Solutions:**
- Decrease anomaly_threshold (e.g., 0.7 → 0.6)
- Check if model is trained
- Verify input data quality

### Issue: Poor Performance Predictions

**Solutions:**
- Increase embedding_dim (e.g., 64 → 128)
- Train for more epochs
- Ensure sufficient training data

---

## References

- **AD4CD GitHub**: https://github.com/BIMK/Intelligent-Education/tree/main/AD4CD
- **SQKT Integration**: `docs/SQKT_INTEGRATION_GUIDE.md`
- **PyTorch**: https://pytorch.org/
- **Neo4j**: https://neo4j.com/docs/

---

**Status**: ✅ **INTEGRATED AND OPERATIONAL**  
**Version**: 2.0.0  
**Model**: AD4CD (Anomaly Detection for Cognitive Diagnosis)  
**Integration**: SQKT + Neo4j + G-CDM

