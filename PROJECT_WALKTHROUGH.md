# 🎓 KTCD_Aug - Complete Project Walkthrough

**Version**: 4.0.0  
**Date**: November 4, 2025  
**Purpose**: Comprehensive guide to understanding and using the KTCD_Aug platform

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [AI Models Explained](#ai-models-explained)
4. [Docker Setup](#docker-setup)
5. [Development Workflow](#development-workflow)
6. [API Documentation](#api-documentation)
7. [Database Schema](#database-schema)
8. [Deployment Guide](#deployment-guide)

---

## 🌟 Project Overview

### What is KTCD_Aug?

KTCD_Aug (Knowledge Tracing & Cognitive Diagnosis - Augmented) is an intelligent educational platform that uses state-of-the-art AI models to provide personalized learning experiences.

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    KTCD_Aug Platform                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Knowledge Tracing (SQKT + MLFBK)                        │
│     → Tracks student learning over time                     │
│     → Predicts future performance                           │
│     → Identifies skill gaps                                 │
│                                                              │
│  2. Cognitive Diagnosis (G-CDM + AD4CD)                     │
│     → Assesses concept-level mastery                        │
│     → Detects anomalies (cheating, guessing)                │
│     → Provides diagnostic insights                          │
│                                                              │
│  3. Recommendation System (RL Agent)                        │
│     → Suggests next best learning action                    │
│     → Personalizes content based on hobbies                 │
│     → Adapts difficulty dynamically                         │
│                                                              │
│  4. Personalized Lab Generation                             │
│     → Combines outputs from all 3 AI models                 │
│     → Creates custom exercises                              │
│     → Adapts to student needs                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                          │
│  HTML Templates + JavaScript + Chart.js + Tailwind CSS      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Application Layer                          │
│  Flask Web Server (nexus_app.py)                            │
│  ├── Routes (4 blueprints)                                  │
│  ├── Templates (20+ HTML files)                             │
│  └── Static Assets (CSS, JS, images)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Services Layer                             │
│  ├── Knowledge Tracing (SQKT + MLFBK)                       │
│  ├── Cognitive Diagnosis (G-CDM + AD4CD)                    │
│  ├── Recommendation (RL Agent)                              │
│  ├── Content Generation (Labs, Quizzes, Blogs)             │
│  ├── Knowledge Graph (Neo4j integration)                    │
│  ├── Assessment (Grading, evaluation)                       │
│  ├── Authentication (bcrypt)                                │
│  └── Chatbot (AI assistant)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                    Data Layer                                │
│  Neo4j Graph Database                                        │
│  ├── 199 Nodes (Class, Topic, Concept, Student, etc.)      │
│  ├── 353 Relationships (INCLUDES, KNOWS, etc.)             │
│  └── Properties (mastery_level, scores, etc.)              │
└─────────────────────────────────────────────────────────────┘
```

### Microservices Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web App    │────▶│  AI Models   │────▶│   Neo4j DB   │
│  Port 8080   │     │  Port 5000   │     │  Port 7687   │
└──────────────┘     └──────────────┘     └──────────────┘
      │                     │                     │
      └─────────────────────┴─────────────────────┘
                    ktcd_network
```

---

## 🤖 AI Models Explained

### 1. Knowledge Tracing (SQKT + MLFBK)

**Purpose**: Track student knowledge state over time

**How it works**:
```python
# Student takes quiz
student_response = {
    'student_id': 'student_123',
    'question_id': 'q_456',
    'response': 'correct',
    'time_spent': 45  # seconds
}

# SQKT processes sequence
knowledge_state = sqkt.update_state(student_response)
# → [0.72, 0.45, 0.89, ...]  # Vector of concept masteries

# Predict next performance
prediction = sqkt.predict_next('student_123', 'q_789')
# → 0.68  # 68% chance of correct answer
```

**Outputs**:
- Knowledge state vector (128-dim)
- Concept-level predictions (0-1 scale)
- Learning trajectory (velocity, consistency)
- Skill gaps (concepts needing practice)

### 2. Cognitive Diagnosis (G-CDM + AD4CD)

**Purpose**: Assess concept mastery and detect anomalies

**How it works**:
```python
# Diagnose student response
diagnosis = ad4cd.diagnose_response(
    student_id='student_123',
    question_id='q_456',
    response='correct'
)

# Result
{
    'mastery_level': 0.75,
    'confidence': 0.85,
    'anomaly_detected': False,
    'category': 'proficient'
}
```

**Outputs**:
- Mastery profile (per concept)
- Anomaly detection (cheating, guessing)
- Diagnostic insights (strengths, weaknesses)
- Confidence scores

### 3. Recommendation System (RL Agent)

**Purpose**: Suggest optimal next learning action

**How it works**:
```python
# Get recommendation
recommendation = rl_agent.get_recommendation(
    student_id='student_123',
    current_topic='NoSQL Databases'
)

# Result
{
    'action': 'exercise',
    'concept': 'CAP Theorem',
    'difficulty': 'medium',
    'learning_gain': 0.18,
    'personalization': 'game leaderboard context'
}
```

**Outputs**:
- Next best action (exercise, explanation, hint)
- Content recommendations (ranked by value)
- Difficulty adjustment
- Hobby-based personalization

---

## 🐳 Docker Setup

### Container Configuration

#### **1. Neo4j Database**
```yaml
neo4j:
  image: neo4j:5.15.0
  ports:
    - "7474:7474"  # Browser
    - "7687:7687"  # Bolt
  environment:
    - NEO4J_AUTH=neo4j/ktcd_password123
    - NEO4J_PLUGINS=["apoc", "graph-data-science"]
  volumes:
    - neo4j_data:/data
```

#### **2. AI Models Service**
```yaml
ai-models:
  build:
    dockerfile: Dockerfile.ai-models
  ports:
    - "5000:5000"
  environment:
    - NEO4J_URI=bolt://neo4j:7687
  depends_on:
    - neo4j
```

#### **3. Web Application**
```yaml
web:
  build:
    dockerfile: Dockerfile
  ports:
    - "8080:8080"
  environment:
    - AI_MODELS_URL=http://ai-models:5000
  depends_on:
    - neo4j
    - ai-models
```

### Starting Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f web

# Stop services
docker-compose down
```

---

## 💻 Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/KTCD_Aug.git
cd KTCD_Aug

# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start Development Server

```bash
# Option 1: With Docker
make run

# Option 2: Without Docker
# Terminal 1: Start Neo4j
docker-compose up -d neo4j

# Terminal 2: Start AI models
python ai_models_server.py

# Terminal 3: Start web app
python nexus_app.py
```

### 3. Make Changes

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes
# ... edit files ...

# Test changes
make test

# Format code
make format

# Commit
git add .
git commit -m "feat: add my feature"
```

### 4. Submit Pull Request

```bash
# Push to GitHub
git push origin feature/my-feature

# Open PR on GitHub
```

---

## 📡 API Documentation

### AI Models API Endpoints

#### **Knowledge Tracing**

```bash
# Predict performance
POST http://localhost:5000/api/kt/predict
{
    "student_id": "student_123",
    "question_id": "q_456"
}

# Get concept predictions
POST http://localhost:5000/api/kt/concept-predictions
{
    "student_id": "student_123",
    "concepts": ["CAP Theorem", "Data Modeling"]
}

# Get skill gaps
POST http://localhost:5000/api/kt/skill-gaps
{
    "student_id": "student_123",
    "concepts": ["CAP Theorem"],
    "target_mastery": 0.7
}
```

#### **Cognitive Diagnosis**

```bash
# Diagnose response
POST http://localhost:5000/api/cd/diagnose
{
    "student_id": "student_123",
    "question_id": "q_456",
    "response": "correct"
}

# Get mastery profile
POST http://localhost:5000/api/cd/mastery-profile
{
    "student_id": "student_123",
    "concepts": ["CAP Theorem"]
}
```

#### **Recommendations**

```bash
# Get recommendation
POST http://localhost:5000/api/rec/recommend
{
    "student_id": "student_123",
    "current_topic": "NoSQL Databases"
}
```

---

## 🗄️ Database Schema

### Neo4j Graph Structure

```cypher
// Nodes
(Class)
(Topic)
(Theory)
(Concept)
(Student)
(Teacher)
(Video)
(Lab)
(Quiz)
(Question)
(ReadingMaterial)

// Relationships
(Class)-[:INCLUDES]->(Topic)
(Topic)-[:HAS_THEORY]->(Theory)
(Theory)-[:CONSISTS_OF]->(Concept)
(Student)-[:REGISTERED_IN]->(Class)
(Student)-[:KNOWS {mastery_level}]->(Concept)
(Video)-[:EXPLAINS]->(Theory)
(Lab)-[:PRACTICES]->(Topic)
(Lab)-[:APPLIES]->(Concept)
(Quiz)-[:TESTS]->(Concept)
(Question)-[:TESTS]->(Concept)
(ReadingMaterial)-[:EXPLAINS]->(Concept)
```

### Example Queries

```cypher
// Get student's mastery profile
MATCH (s:Student {student_id: 'student_123'})-[k:KNOWS]->(c:Concept)
RETURN c.name, k.mastery_level
ORDER BY k.mastery_level ASC

// Get topics in a class
MATCH (class:Class {name: 'Big Data Analysis'})-[:INCLUDES]->(t:Topic)
RETURN t.name, t.order
ORDER BY t.order

// Get concepts needing practice
MATCH (s:Student {student_id: 'student_123'})-[k:KNOWS]->(c:Concept)
WHERE k.mastery_level < 0.7
RETURN c.name, k.mastery_level
ORDER BY k.mastery_level ASC
LIMIT 5
```

---

## 🚀 Deployment Guide

### Production Deployment

#### **1. Prepare Environment**

```bash
# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=False

# Update .env
SECRET_KEY=<strong-random-key>
NEO4J_PASSWORD=<strong-password>
```

#### **2. Build Docker Images**

```bash
# Build images
docker-compose build

# Tag for registry
docker tag ktcd_aug:latest your-registry/ktcd_aug:4.0.0
docker tag ktcd_aug_ai:latest your-registry/ktcd_aug_ai:4.0.0

# Push to registry
docker push your-registry/ktcd_aug:4.0.0
docker push your-registry/ktcd_aug_ai:4.0.0
```

#### **3. Deploy**

```bash
# On production server
docker-compose -f docker-compose.prod.yml up -d

# Check health
curl http://localhost:8080/health
curl http://localhost:5000/health
```

---

## 📊 Monitoring

### Health Checks

```bash
# Web app
curl http://localhost:8080/health

# AI models
curl http://localhost:5000/health

# Neo4j
docker exec ktcd_neo4j cypher-shell -u neo4j -p ktcd_password123 "RETURN 1"
```

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f web
docker-compose logs -f ai-models
docker-compose logs -f neo4j
```

---

## 🎯 Common Tasks

### Add New Student

```python
from services.auth.student_auth import get_student_auth_service

auth = get_student_auth_service()
student = auth.register_student(
    email='new@example.com',
    password='password123',
    name='New Student',
    class_id='class_123'
)
```

### Generate Personalized Lab

```python
from services.content_generation.services.personalized_lab_generator import PersonalizedLabGenerator

generator = PersonalizedLabGenerator(sqkt, ad4cd, rl_agent, lab_gen)
lab = generator.generate_personalized_lab(
    student_id='student_123',
    topic_name='NoSQL Databases',
    concepts=['CAP Theorem', 'Data Modeling']
)
```

---

**For more details, see**:
- [ULTIMATE_PROJECT_SUMMARY.md](ULTIMATE_PROJECT_SUMMARY.md) - Complete technical documentation
- [LAB_GENERATION_INPUTS_OUTPUTS.md](docs/LAB_GENERATION_INPUTS_OUTPUTS.md) - Lab generation guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

---

**Made with ❤️ by the KTCD_Aug Team**

