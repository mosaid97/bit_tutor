# Mathematical Equations Summary - KTCD Platform

## Quick Reference Guide for All AI Models

---

## 1. MLFBK (Multi-Layer Feature-Based Knowledge Tracing)

### Core Equations

**Feature Fusion:**
```
x_t = e_item + e_skill + e_response + e_type + e_code + e_question + e_pos
```

**Multi-Head Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

where:
- Q = x_t W_Q  (Query matrix)
- K = x_t W_K  (Key matrix)
- V = x_t W_V  (Value matrix)
- d_k = dimension of key vectors
```

**Transformer Layer:**
```
h'_t = LayerNorm(x_t + MultiHeadAttention(x_t))
h_t = LayerNorm(h'_t + FeedForward(h'_t))
```

**Prediction:**
```
P(correct_{t+1}) = σ(W_o × h_t + b_o)

where σ(z) = 1 / (1 + e^(-z))
```

**Loss Function:**
```
L = -1/N ∑[y_t log(P_t) + (1 - y_t)log(1 - P_t)]
```

---

## 2. GNN-CDM (Graph Neural Network - Cognitive Diagnosis Model)

### Core Equations

**Graph Convolution:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

where:
- Ã = A + I (adjacency matrix with self-loops)
- D̃ = degree matrix
- H^(l) = node features at layer l
- W^(l) = learnable weights
- σ = ReLU activation
```

**Mastery Prediction:**
```
m̂_k = σ(MLP([h_s || h_k]))

Expanded:
m̂_k = σ(W_2 × ReLU(W_1 × [h_s || h_k] + b_1) + b_2)

where:
- h_s = student embedding
- h_k = knowledge component embedding
- || = concatenation
```

### Guessing & Slipping Parameters

**Guessing (g):**
```
g = P(correct | not mastered) = 0.25

Interpretation: 25% chance of guessing correctly on 4-option multiple choice
```

**Slipping (s):**
```
s = P(incorrect | mastered) = 0.15

Interpretation: 15% chance of careless error despite mastery
```

**Item Response Function:**
```
P(correct | mastery = m) = g + (1 - g - s) × m

Special cases:
- If m = 0: P(correct) = 0.25 (pure guessing)
- If m = 1: P(correct) = 0.85 (1 - slipping)
```

**Inverse - Estimating Mastery:**
```
m̂ = (accuracy - g) / (1 - g - s)

Example:
accuracy = 0.70, g = 0.25, s = 0.15
m̂ = (0.70 - 0.25) / (1 - 0.25 - 0.15)
m̂ = 0.45 / 0.60 = 0.75 (75% mastery)
```

**Bayesian Knowledge Tracing (BKT):**
```
P(L_t | correct) = [P(L_{t-1})(1-s)] / [P(L_{t-1})(1-s) + (1-P(L_{t-1}))g]

P(L_t | incorrect) = [P(L_{t-1})s] / [P(L_{t-1})s + (1-P(L_{t-1}))(1-g)]
```

---

## 3. RL Recommender Agent (Q-Learning)

### Core Equations

**State Discretization:**
```
s_t = (⌊10m_1⌋, ⌊10m_2⌋, ..., ⌊10m_K⌋)

where m_k ∈ [0,1] is continuous mastery
```

**ε-Greedy Policy:**
```
a_t = {
    random action,              with probability ε
    argmax_a Q(s_t, a),        with probability 1-ε
}
```

**Exploration Decay:**
```
ε_t = ε_min + (ε_max - ε_min) × e^(-λt)

Parameters:
- ε_max = 1.0
- ε_min = 0.01
- λ = 0.01
```

**Q-Learning Update (Bellman Equation):**
```
Q(s_t, a_t) ← Q(s_t, a_t) + α[r_t + γ max_a' Q(s_{t+1}, a') - Q(s_t, a_t)]

Parameters:
- α = 0.1 (learning rate)
- γ = 0.9 (discount factor)
```

**Temporal Difference Error:**
```
δ_t = r_t + γ max_a' Q(s_{t+1}, a') - Q(s_t, a_t)
```

**Reward Function:**
```
r_t = base_reward + improvement_bonus + mastery_bonus

base_reward = {
    +1.0,  if correct and m_k < 0.8
    +0.5,  if correct and m_k ≥ 0.8
    -0.5,  if incorrect and m_k < 0.5
    -1.0,  if incorrect and m_k ≥ 0.5
}

improvement_bonus = 0.5 × (m_k^new - m_k^old)

mastery_bonus = {
    +1.0,  if m_k^new ≥ 0.8 and m_k^old < 0.8
    0,     otherwise
}
```

---

## Hyperparameters

### MLFBK
| Parameter | Value | Description |
|-----------|-------|-------------|
| embedding_dim | 64 | Dimension of embeddings |
| num_heads | 8 | Attention heads |
| num_layers | 4 | Transformer layers |
| max_seq_len | 100 | Max sequence length |
| learning_rate | 0.001 | Adam optimizer LR |

### GNN-CDM
| Parameter | Value | Description |
|-----------|-------|-------------|
| embedding_dim | 64 | GNN embedding size |
| num_gcn_layers | 2 | Graph conv layers |
| guessing_param | 0.25 | Guessing probability |
| slipping_param | 0.15 | Slipping probability |
| learning_rate | 0.01 | Optimizer LR |

### RL Agent
| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.1 | Learning rate |
| γ (gamma) | 0.9 | Discount factor |
| ε_max | 1.0 | Max exploration |
| ε_min | 0.01 | Min exploration |
| λ (lambda) | 0.01 | Decay rate |

---

## Performance Metrics

### MLFBK
```
Accuracy = (TP + TN) / Total
AUC-ROC = Area Under ROC Curve
```

### GNN-CDM
```
RMSE = √(1/N ∑(m - m̂)²)
MAE = 1/N ∑|m - m̂|
```

### RL Agent
```
Cumulative Reward = ∑ r_t
Learning Efficiency = Mastery Gain / Exercises
```

---

## Example Calculations

### Example 1: Mastery from Accuracy
```
Given:
- Student answers 7/10 questions correctly
- Guessing parameter g = 0.25
- Slipping parameter s = 0.15

Calculate mastery:
accuracy = 7/10 = 0.70
m̂ = (0.70 - 0.25) / (1 - 0.25 - 0.15)
m̂ = 0.45 / 0.60
m̂ = 0.75 (75% mastery)
```

### Example 2: Q-Learning Update
```
Given:
- Current state: s = (7, 5, 8)
- Action taken: a = 2
- Current Q-value: Q(s,a) = 0.65
- Reward received: r = +1.0
- Next state: s' = (7, 6, 8)
- Max Q-value in s': max Q(s',a') = 0.78
- α = 0.1, γ = 0.9

Calculate new Q-value:
TD_target = r + γ × max Q(s',a')
TD_target = 1.0 + 0.9 × 0.78 = 1.702

TD_error = TD_target - Q(s,a)
TD_error = 1.702 - 0.65 = 1.052

Q_new(s,a) = Q(s,a) + α × TD_error
Q_new(s,a) = 0.65 + 0.1 × 1.052
Q_new(s,a) = 0.7552
```

### Example 3: Probability with Guessing/Slipping
```
Given:
- Mastery level m = 0.60
- Guessing g = 0.25
- Slipping s = 0.15

Calculate probability of correct answer:
P(correct) = g + (1 - g - s) × m
P(correct) = 0.25 + (1 - 0.25 - 0.15) × 0.60
P(correct) = 0.25 + 0.60 × 0.60
P(correct) = 0.25 + 0.36
P(correct) = 0.61 (61% chance)
```

---

**Document Version:** 1.0  
**Last Updated:** October 22, 2025  
**For:** KTCD Educational Platform

