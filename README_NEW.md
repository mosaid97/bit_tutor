# 🎓 KTCD_Aug - Knowledge Tracing & Cognitive Diagnosis Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.15-blue.svg)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An intelligent educational platform that combines Knowledge Tracing, Cognitive Diagnosis, and Personalized Recommendations to create adaptive learning experiences powered by state-of-the-art AI models.**

---

## 🌟 Key Features

### 🤖 **AI-Powered Learning**
- **SQKT + MLFBK**: Sequential Knowledge Tracing with Multi-Feature BERT
- **G-CDM + AD4CD**: Graph-based Cognitive Diagnosis with Anomaly Detection
- **RL Agent**: Reinforcement Learning for Personalized Recommendations
- **Personalized Labs**: AI-generated exercises based on student mastery and hobbies

### 📚 **For Students**
- ✅ Personalized learning paths based on cognitive diagnosis
- ✅ Interactive Jupyter-style coding labs
- ✅ Video lectures with concept explanations
- ✅ Graded quizzes and assessments
- ✅ Real-time progress tracking with spider web visualization
- ✅ 24/7 AI chatbot assistance
- ✅ Hobby-based personalization (gaming, music, sports contexts)

### 👨‍🏫 **For Educators**
- ✅ Knowledge graph-based content organization
- ✅ Student performance analytics and insights
- ✅ Adaptive assessment generation
- ✅ Anomaly detection (cheating, guessing)
- ✅ Concept-level mastery tracking

---

## 🚀 Quick Start

### **Prerequisites**
- Docker & Docker Compose
- Python 3.12+
- 4GB+ RAM

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/KTCD_Aug.git
cd KTCD_Aug
```

### **2. Set Up Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key (optional)
nano .env
```

### **3. Start with Docker (Recommended)**
```bash
# Start all services (Neo4j + AI Models + Web App)
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f
```

### **4. Access the Platform**
```
🌐 Web Application: http://localhost:8080
📊 Neo4j Browser: http://localhost:7474
🤖 AI Models API: http://localhost:5000

Test Student Login:
Email: roma@example.com
Password: roma123
```

### **5. Stop Services**
```bash
docker-compose down
```

---

## 📦 Installation (Without Docker)

### **1. Install Dependencies**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### **2. Start Neo4j**
```bash
# Using Docker for Neo4j only
docker-compose up -d neo4j

# Or install Neo4j locally and start it
```

### **3. Run the Application**
```bash
# Start AI models server (in one terminal)
python ai_models_server.py

# Start web application (in another terminal)
python nexus_app.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     KTCD_Aug Platform                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Web App    │  │  AI Models   │  │   Neo4j DB   │     │
│  │  (Flask)     │  │   Service    │  │  (Graph DB)  │     │
│  │  Port 8080   │  │  Port 5000   │  │  Port 7687   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              AI Models Layer                          │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • SQKT + MLFBK (Knowledge Tracing)                  │  │
│  │  • G-CDM + AD4CD (Cognitive Diagnosis)               │  │
│  │  • RL Agent (Recommendations)                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Knowledge Graph (Neo4j)                     │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • 199 Nodes (Class, Topic, Concept, Student, etc.)  │  │
│  │  • 353 Relationships (INCLUDES, KNOWS, etc.)         │  │
│  │  • 47 Concepts across 5 Topics                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.12, Flask 3.1 |
| **Database** | Neo4j 5.15 (Graph Database) |
| **AI/ML** | PyTorch 2.8, Transformers, torch-geometric |
| **Frontend** | HTML5, JavaScript, Tailwind CSS, Chart.js |
| **Containerization** | Docker, Docker Compose |
| **APIs** | OpenAI GPT-4 (optional) |
| **Security** | bcrypt, python-dotenv |

---

## 📁 Project Structure

```
KTCD_Aug/
├── 📄 nexus_app.py              # Main Flask application
├── 📄 ai_models_server.py       # AI models microservice
├── 📄 docker-compose.yml        # Docker orchestration
├── 📄 Dockerfile                # Web app container
├── 📄 Dockerfile.ai-models      # AI models container
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env.example              # Environment template
├── 📄 .gitignore                # Git ignore rules
│
├── 📁 routes/                   # Flask blueprints
│   ├── student_portal_routes.py
│   ├── student_learning_routes.py
│   ├── student_portfolio_routes.py
│   └── student_registration_routes.py
│
├── 📁 services/                 # Business logic
│   ├── knowledge_tracing/       # SQKT + MLFBK
│   ├── cognitive_diagnosis/     # G-CDM + AD4CD
│   ├── recommendation/          # RL Agent
│   ├── content_generation/      # Lab/Quiz generation
│   ├── knowledge_graph/         # Neo4j integration
│   ├── assessment/              # Assessment logic
│   ├── auth/                    # Authentication
│   └── chatbot/                 # AI chatbot
│
├── 📁 templates/                # HTML templates
│   ├── student/                 # Student pages
│   └── teacher/                 # Teacher pages
│
├── 📁 static/                   # Static assets
│   ├── css/                     # Stylesheets
│   ├── js/                      # JavaScript
│   └── images/                  # Images
│
├── 📁 utilities/                # Utility scripts
│   ├── benchmark_algorithms.py  # AI model benchmarking
│   ├── visualize_knowledge_graph.py
│   ├── cleanup_knowledge_graph.py
│   └── setup_demo_system.py
│
├── 📁 docs/                     # Documentation
│   ├── ULTIMATE_PROJECT_SUMMARY.md
│   ├── LAB_GENERATION_INPUTS_OUTPUTS.md
│   ├── AI_MODELS_FOR_LAB_GENERATION_SUMMARY.md
│   └── ... (15+ technical docs)
│
├── 📁 lab_tutor/                # Lab content
│   └── knowledge_graph_builder/
│
├── 📁 data/                     # Data files
│   └── generated_blogs/
│
├── 📁 models/                   # AI model checkpoints
│   └── checkpoints/
│
└── 📁 logs/                     # Application logs
```

---

## 🔧 Configuration

### **Environment Variables**

Key variables in `.env`:

```bash
# OpenAI API (optional, for LLM features)
OPENAI_API_KEY=your-api-key-here

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ktcd_password123

# AI Models
SQKT_MODEL_PATH=models/sqkt_model.pt
GCDM_MODEL_PATH=models/gcdm_model.pt
RL_MODEL_PATH=models/rl_agent.pt

# Application
FLASK_ENV=production
SECRET_KEY=your-secret-key
```

See `.env.example` for all available options.

---

## 🧪 Testing

### **Run Benchmarks**
```bash
python utilities/benchmark_algorithms.py
```

### **Verify Pipelines**
```bash
python utilities/verify_and_test_pipelines.py
```

### **Visualize Knowledge Graph**
```bash
python utilities/visualize_knowledge_graph.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ULTIMATE_PROJECT_SUMMARY.md](ULTIMATE_PROJECT_SUMMARY.md) | Complete project overview |
| [LAB_GENERATION_INPUTS_OUTPUTS.md](docs/LAB_GENERATION_INPUTS_OUTPUTS.md) | Lab generation guide |
| [AI_MODELS_FOR_LAB_GENERATION_SUMMARY.md](docs/AI_MODELS_FOR_LAB_GENERATION_SUMMARY.md) | AI models integration |
| [COMPARISON_WITH_RECENT_MODELS.md](COMPARISON_WITH_RECENT_MODELS.md) | Benchmark results |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Neo4j** for the graph database platform
- **PyTorch** for deep learning framework
- **OpenAI** for GPT-4 API
- **Hugging Face** for transformer models

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ by the KTCD_Aug Team**

