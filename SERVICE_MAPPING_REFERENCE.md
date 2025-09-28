# 🗺️ BIT Tutor - Service Mapping Reference

## 📋 **Overview**
This document provides a detailed mapping of current code files to the new reorganized structure, including specific classes, functions, and line numbers.

---

## 🧠 **Core AI Services Mapping**

### **1. Knowledge Graph Service**
**Source File:** `knowledge_graph.py` (1,278 lines)

#### **Target Structure:**
```
services/knowledge_graph/
├── models/
│   ├── student_knowledge_graph.py     # Lines 43-965
│   └── cognitive_foundation.py        # Lines 966-1278
├── repositories/
│   ├── neo4j_repository.py           # Lines 137-196 (Neo4j methods)
│   └── local_graph_repository.py     # Lines 198-340 (Local graph methods)
├── services/
│   ├── graph_builder_service.py      # Lines 966-1055 (build_cognitive_foundation)
│   └── mastery_tracking_service.py   # Lines 343-604 (mastery methods)
└── utils/
    └── graph_utilities.py            # Lines 1-42, helper functions
```

#### **Key Classes & Functions:**
- **`StudentKnowledgeGraph`** (Lines 43-965) → `models/student_knowledge_graph.py`
- **`build_cognitive_foundation()`** (Lines 966-1055) → `models/cognitive_foundation.py`
- **`_build_default_foundation()`** (Lines 1056-1078) → `models/cognitive_foundation.py`
- **`run_educational_agent()`** (Lines 1081-1278) → `services/educational_agent/`

---

### **2. Knowledge Tracing Service**
**Source File:** `knowledge_tracing.py` (200+ lines)

#### **Target Structure:**
```
services/knowledge_tracing/
├── models/
│   ├── llm_skill_extractor.py        # Lines 7-27 (LLM_Skill_Extractor)
│   ├── astnn_model.py                 # Lines 29-46 (ASTNN)
│   ├── text_embedder.py               # Lines 47-57 (TextEmbedder)
│   └── mlfbk_model.py                 # Lines 59-200+ (MLFBK_Model, MLFBK_KnowledgeTracer)
├── services/
│   ├── skill_extraction_service.py   # Service wrapper for LLM_Skill_Extractor
│   └── knowledge_tracing_service.py  # Service wrapper for MLFBK
└── utils/
    └── tracing_utilities.py          # Helper functions
```

#### **Key Classes:**
- **`LLM_Skill_Extractor`** → `models/llm_skill_extractor.py`
- **`ASTNN`** → `models/astnn_model.py`
- **`TextEmbedder`** → `models/text_embedder.py`
- **`MLFBK_Model`** → `models/mlfbk_model.py`
- **`MLFBK_KnowledgeTracer`** → `models/mlfbk_model.py`

---

### **3. Cognitive Diagnosis Service**
**Source File:** `cognitive_diagnosis.py` (400+ lines)

#### **Target Structure:**
```
services/cognitive_diagnosis/
├── models/
│   ├── cold_start_assessor.py        # Lines 92-196 (LLM_Cold_Start_Assessor)
│   ├── gnn_cdm.py                     # Lines 198-233 (GNN_CDM)
│   └── explainable_ai_engine.py      # Lines 235-400+ (ExplainableAIEngine)
├── services/
│   ├── diagnosis_service.py          # Service orchestration
│   └── explanation_service.py        # XAI service wrapper
└── utils/
    └── diagnosis_utilities.py        # convert_nx_to_pyg, mock classes
```

#### **Key Classes:**
- **`LLM_Cold_Start_Assessor`** → `models/cold_start_assessor.py`
- **`GNN_CDM`** → `models/gnn_cdm.py`
- **`ExplainableAIEngine`** → `models/explainable_ai_engine.py`
- **`convert_nx_to_pyg()`** → `utils/diagnosis_utilities.py`

---

### **4. Recommendation Service**
**Source File:** `recommendation_model.py` (300+ lines)

#### **Target Structure:**
```
services/recommendation/
├── models/
│   ├── content_generator.py          # LLM_Content_Generator class
│   └── rl_agent.py                   # RL_Recommender_Agent class
├── services/
│   ├── recommendation_service.py     # Service orchestration
│   └── personalization_service.py   # Personalization logic
└── utils/
    └── recommendation_utilities.py   # Helper functions
```

---

## 🌐 **Web Application Services Mapping**

### **1. Main Web Application**
**Source File:** `nexus_app.py` (2,500+ lines)

#### **Target Structure:**
```
apps/web_app/
├── app.py                            # Lines 45-67 (Flask app initialization)
├── routes/
│   ├── student_routes.py             # Lines 1904-1928 (student routes)
│   ├── api_routes.py                 # Lines 1929-2002 (API endpoints)
│   ├── chat_routes.py                # Lines 2303-2404 (chat endpoints)
│   └── analytics_routes.py           # Analytics endpoints
├── middleware/
│   └── session_middleware.py         # Session management logic
└── config/
    ├── settings.py                   # Lines 47-54 (app config)
    └── constants.py                  # Global constants
```

#### **Key Routes to Extract:**
- **Student Routes:**
  - `@app.route('/')` → `student_routes.py`
  - `@app.route('/student/<student_id>')` → `student_routes.py`

- **API Routes:**
  - `@app.route('/api/student/<student_id>/live_metrics')` → `api_routes.py`
  - `@app.route('/api/student/<student_id>/personalized_labs')` → `api_routes.py`
  - `@app.route('/api/student/<student_id>/personalized_quizzes')` → `api_routes.py`
  - `@app.route('/api/student/<student_id>/learning_materials')` → `api_routes.py`

- **Chat Routes:**
  - `@app.route('/api/student/<student_id>/chat', methods=['POST'])` → `chat_routes.py`
  - `@app.route('/api/student/<student_id>/chat/history')` → `chat_routes.py`

#### **Key Classes to Extract:**
- **`ComprehensiveBITTutorAI`** (Lines 1348-1879) → `services/educational_agent/models/comprehensive_ai.py`
- **`KnowledgeGraphLearningMaterialsGenerator`** (Lines 410-745) → `content_generation/learning_materials/services/materials_generator.py`
- **`KnowledgeGraphLabGenerator`** (Lines 746-1231) → `content_generation/assessments/services/lab_generator.py`
- **`KnowledgeGraphQuizGenerator`** (Lines 1232-1347) → `content_generation/assessments/services/quiz_generator.py`

---

## 📊 **Data Services Mapping**

### **1. Student Data Service**
**Source File:** `student_data_service.py` (400+ lines)

#### **Target Structure:**
```
data_services/student_data/
├── models/
│   ├── student_profile.py            # Student profile models
│   └── student_analytics.py          # Analytics models
├── repositories/
│   ├── student_repository.py         # Lines 18-374 (StudentDataService)
│   └── analytics_repository.py       # Analytics data persistence
├── services/
│   ├── student_data_service.py       # Service wrapper
│   └── analytics_service.py          # Analytics processing
└── generators/
    └── student_data_generator.py     # From generate_student_data.py
```

### **2. Student Data Generator**
**Source File:** `generate_student_data.py` (500+ lines)

#### **Target Structure:**
```
data_services/student_data/generators/
├── student_data_generator.py         # Main StudentDataGenerator class
└── mock_data_generator.py            # Mock data utilities
```

---

## 📈 **Analytics Services Mapping**

### **1. Web Visualizations**
**Source File:** `web_visualizations.py` (1,500+ lines)

#### **Target Structure:**
```
analytics/visualizations/
├── charts/
│   ├── mastery_charts.py             # Lines 1-300 (mastery-related charts)
│   ├── progress_charts.py            # Lines 301-600 (progress charts)
│   ├── knowledge_graph_viz.py        # Lines 1200-1400 (KG visualizations)
│   └── ai_model_viz.py               # Lines 1400-1500 (AI model viz)
├── services/
│   ├── chart_generator.py            # Chart generation service
│   └── interactive_charts.py         # Interactive chart utilities
└── utils/
    ├── chart_utilities.py            # Helper functions
    └── rendering_utils.py             # Rendering utilities
```

#### **Key Functions to Extract:**
- **Mastery Charts:**
  - `create_mastery_bar_chart()` → `charts/mastery_charts.py`
  - `create_mastery_progress_tracker()` → `charts/mastery_charts.py`

- **Progress Charts:**
  - `create_progress_chart()` → `charts/progress_charts.py`
  - `create_journey_timeline()` → `charts/progress_charts.py`

- **Knowledge Graph Visualizations:**
  - `create_knowledge_graph_visualization()` → `charts/knowledge_graph_viz.py`
  - `create_gnn_message_passing_visualization()` → `charts/knowledge_graph_viz.py`

### **2. Student Dashboard**
**Source File:** `student_dashboard.py` (400+ lines)

#### **Target Structure:**
```
analytics/dashboards/
├── components/
│   ├── student_dashboard.py          # Main dashboard component
│   └── learning_path_recommender.py  # PersonalizedLearningPathRecommender
├── services/
│   ├── dashboard_service.py          # Dashboard orchestration
│   └── metrics_service.py            # Metrics calculation
└── models/
    └── dashboard_models.py           # Dashboard data models
```

---

## 🎨 **Content Generation Services Mapping**

### **1. Themed Content Generator**
**Source File:** `themed_content_generator.py` (430+ lines)

#### **Target Structure:**
```
content_generation/themed_content/
├── models/
│   └── theme_models.py               # Theme data structures
├── services/
│   ├── themed_generator.py           # Lines 1-430 (ThemedContentGenerator)
│   └── personalization_engine.py    # Personalization logic
└── templates/
    ├── quiz_templates.py             # Quiz generation templates
    └── lab_templates.py              # Lab generation templates
```

### **2. Learning Materials Generator**
**Source:** `nexus_app.py` (Lines 410-745)

#### **Target Structure:**
```
content_generation/learning_materials/
├── services/
│   ├── materials_generator.py       # KnowledgeGraphLearningMaterialsGenerator
│   └── source_curator.py           # Content curation logic
├── sources/
│   ├── verified_sources.py         # Educational sources mapping
│   └── content_fetcher.py          # Web content fetching
└── models/
    └── material_models.py          # Learning material models
```

---

## 🤖 **AI Chat Services Mapping**

### **1. Chat Functionality**
**Source:** `nexus_app.py` (Lines 2303-2404)

#### **Target Structure:**
```
ai_chat/
├── models/
│   ├── chatbot_models.py            # Chat models
│   └── conversation_models.py       # Conversation data models
├── services/
│   ├── chat_service.py              # Main chat service
│   ├── conversation_manager.py      # Conversation management
│   └── response_generator.py        # Response generation
├── handlers/
│   ├── intent_handler.py            # Intent recognition
│   └── context_handler.py           # Context management
└── utils/
    ├── nlp_utilities.py             # NLP helper functions
    └── chat_utilities.py            # Chat utilities
```

---

## 🔧 **Shared Utilities Mapping**

### **1. Common Utilities**
**Sources:** Various files

#### **Target Structure:**
```
shared/
├── utils/
│   ├── data_utils.py                # Data processing utilities
│   ├── file_utils.py                # File operations
│   ├── validation_utils.py          # Input validation
│   └── logging_utils.py             # Logging configuration
├── models/
│   ├── base_models.py               # Base model classes
│   └── common_models.py             # Shared data models
├── exceptions/
│   ├── service_exceptions.py        # Service-specific exceptions
│   └── validation_exceptions.py     # Validation exceptions
├── constants/
│   ├── app_constants.py             # Application constants
│   └── ai_constants.py              # AI model constants
└── decorators/
    ├── auth_decorators.py           # Authentication decorators
    └── validation_decorators.py     # Validation decorators
```

---

## 🧪 **Testing Structure Mapping**

### **1. Test Organization**
**Current Files:** `test_*.py`

#### **Target Structure:**
```
tests/
├── unit/
│   ├── services/
│   │   ├── test_knowledge_graph_service.py
│   │   ├── test_knowledge_tracing_service.py
│   │   ├── test_cognitive_diagnosis_service.py
│   │   └── test_recommendation_service.py
│   ├── models/
│   │   ├── test_student_models.py
│   │   └── test_ai_models.py
│   └── utils/
│       └── test_utilities.py
├── integration/
│   ├── api_tests/
│   │   ├── test_student_api.py      # From test_api.py
│   │   └── test_chat_api.py
│   └── service_tests/
│       └── test_service_integration.py
├── e2e/
│   └── test_complete_workflow.py    # From test_educational_agent.py
└── fixtures/
    ├── student_fixtures.py          # Test data
    └── ai_model_fixtures.py         # AI model test data
```

---

## 📊 **Migration Priority Matrix**

### **High Priority (Week 1)**
1. **Knowledge Graph Service** - Core dependency for all other services
2. **Student Data Service** - Required for web application
3. **Shared Utilities** - Used across all services

### **Medium Priority (Week 2-3)**
1. **Knowledge Tracing Service** - AI functionality
2. **Cognitive Diagnosis Service** - AI functionality
3. **Web Application Routes** - User interface

### **Low Priority (Week 4-5)**
1. **Content Generation Services** - Feature enhancement
2. **Analytics Services** - Visualization features
3. **Testing & Documentation** - Quality assurance

---

## 🎯 **Success Validation**

### **Functional Tests**
- [ ] All existing API endpoints work correctly
- [ ] Student data persistence functions properly
- [ ] AI services produce expected outputs
- [ ] Web interface displays correctly

### **Performance Tests**
- [ ] Response times remain under 200ms
- [ ] Memory usage doesn't increase significantly
- [ ] Startup time improves or remains same

### **Code Quality Tests**
- [ ] All imports resolve correctly
- [ ] No circular dependencies
- [ ] Test coverage maintained or improved
- [ ] Code complexity reduced

---

This mapping reference provides the exact blueprint for migrating each piece of code to its new location in the reorganized structure, ensuring nothing is lost during the transformation process.
