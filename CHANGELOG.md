# Changelog

All notable changes to KTCD_Aug will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker support for AI models microservice
- Comprehensive lab generation with AI model integration
- GitHub-ready project structure
- CI/CD configuration files
- Comprehensive documentation

## [4.0.0] - 2025-11-04

### Added
- **Personalized Lab Generation System**
  - Integration with SQKT + MLFBK for knowledge tracing
  - Integration with G-CDM + AD4CD for cognitive diagnosis
  - Integration with RL Agent for recommendations
  - Hobby-based personalization (gaming, music, sports contexts)
  - Adaptive difficulty and scaffolding
  - Comprehensive documentation (4 new docs)

- **AI Models Comparison**
  - Benchmarking against recent SOTA models (2015-2024)
  - Comparison with DKT, SAKT, SAINT, simpleKT (Knowledge Tracing)
  - Comparison with NCD, NCDM, KaNCD, RCD (Cognitive Diagnosis)
  - Performance metrics and expected improvements

- **Project Organization**
  - Consolidated utils/ and utilities/ folders
  - Cleaned up 35+ unnecessary documentation files
  - Organized 13 utility scripts into 5 subpackages
  - Enhanced utilities/__init__.py with comprehensive documentation

### Changed
- Updated ULTIMATE_PROJECT_SUMMARY.md to version 4.0
- Improved knowledge graph structure (reduced redundancy)
- Enhanced student progress tracking
- Optimized database queries

### Fixed
- Circular import issues in services
- Neo4j connection stability
- Password hashing with bcrypt
- Session management

## [3.0.0] - 2025-10-14

### Added
- **Knowledge Graph Reorganization**
  - Comprehensive labs (1 per topic covering all concepts)
  - Grades stored as student attributes (not separate nodes)
  - Removed redundant relationships
  - Optimized graph structure (36% fewer nodes)

- **Dynamic Student Registration**
  - Students created automatically on registration
  - Password hashing with bcrypt
  - Email uniqueness validation
  - Grades initialization

- **Enhanced Progress Tracking**
  - Spider web visualization for skill mastery
  - Real-time analytics dashboard
  - Concept-level mastery tracking
  - Learning velocity metrics

### Changed
- Reduced knowledge graph from 310 to 199 nodes
- Reduced relationships from 935 to 353
- Improved query performance by 50%

### Removed
- Grade and StudentScore nodes (moved to attributes)
- QuestionBank nodes (integrated into Question nodes)
- Duplicate student data

## [2.0.0] - 2025-10-13

### Added
- **AI Models Integration**
  - SQKT (Sequential Question-based Knowledge Tracing)
  - MLFBK (Multi-Features with Latent Relations BERT KT)
  - G-CDM (Graph-based Cognitive Diagnosis Model)
  - AD4CD (Anomaly Detection for Cognitive Diagnosis)
  - RL Recommendation Agent

- **Personalized Learning Features**
  - Pre-topic assessments
  - Adaptive learning paths
  - Hobby-based content personalization
  - AI chatbot integration

- **Student Dashboard**
  - Progress visualization
  - Grades tracking
  - Cognitive profile display
  - Performance trends

### Changed
- Migrated from IRT to G-CDM for cognitive diagnosis
- Enhanced knowledge tracing with BERT-based features
- Improved recommendation system with RL

## [1.0.0] - 2025-10-12

### Added
- **Core Platform Features**
  - Student registration and authentication
  - Class selection and enrollment
  - Topic browsing and learning
  - Video lectures
  - Reading materials
  - Interactive labs
  - Graded quizzes

- **Knowledge Graph**
  - Neo4j integration
  - Class, Topic, Theory, Concept structure
  - Student progress tracking
  - Content relationships

- **Teacher Portal**
  - Student management
  - Performance analytics
  - Content organization

### Infrastructure
- Flask web framework
- Neo4j graph database
- Docker support for Neo4j
- Basic authentication system

---

## Version History

- **v4.0.0** (2025-11-04) - Personalized Lab Generation & AI Comparison
- **v3.0.0** (2025-10-14) - Knowledge Graph Optimization
- **v2.0.0** (2025-10-13) - AI Models Integration
- **v1.0.0** (2025-10-12) - Initial Release

---

## Upgrade Guide

### From 3.x to 4.x

1. Update dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. No database migration needed (backward compatible)

3. New features available:
   - Personalized lab generation
   - AI model benchmarking
   - Enhanced documentation

### From 2.x to 3.x

1. **Database Migration Required**:
   ```bash
   python utilities/migrate_grades_to_attributes.py
   ```

2. Update environment variables in `.env`

3. Restart services:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### From 1.x to 2.x

1. Install new AI dependencies:
   ```bash
   pip install torch transformers torch-geometric
   ```

2. Initialize AI models:
   ```bash
   python utilities/setup_ai_models.py
   ```

3. Update Neo4j schema:
   ```bash
   python utilities/update_schema_v2.py
   ```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

