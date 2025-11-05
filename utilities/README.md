# KTCD Utility Scripts

Maintenance and management scripts for the KTCD knowledge graph and learning platform.

## Available Scripts

### Knowledge Graph Management

#### `cleanup_knowledge_graph.py`
Clean up unused nodes and duplicate data from the knowledge graph.

**Usage:**
```bash
python3 utilities/cleanup_knowledge_graph.py
```

**Features:**
- Remove duplicate student nodes
- Delete unused nodes
- Verify graph integrity
- Display statistics

---

#### `visualize_knowledge_graph.py`
Visualize and analyze the complete knowledge graph structure.

**Usage:**
```bash
python3 utilities/visualize_knowledge_graph.py
```

**Features:**
- Node and relationship statistics
- Class and topic structure
- Student data overview
- Mermaid diagram export

---

### Knowledge Graph Updates

#### `update_from_lab_tutor_1.py`
Update topics and concepts from lab_tutor-1 extraction files.

**Usage:**
```bash
python3 utilities/update_from_lab_tutor_1.py
```

**Features:**
- Reads JSON extraction files from lab_tutor-1
- Creates Topic, Theory, and Concept nodes
- Establishes proper relationships
- Links to Big Data Analysis class

---

#### `create_comprehensive_labs.py`
Generate comprehensive Jupyter-style labs for all topics.

**Usage:**
```bash
python3 utilities/create_comprehensive_labs.py
```

**Features:**
- Creates one lab per topic
- Covers all concepts in the topic
- Generates interactive coding cells
- Stores labs in Neo4j

---

#### `run_knowledge_graph_update.py`
Master script to run complete knowledge graph update.

**Usage:**
```bash
python3 utilities/run_knowledge_graph_update.py
```

**Features:**
- Runs update_from_lab_tutor_1.py
- Runs create_comprehensive_labs.py
- Verifies update success
- Provides statistics

---

## Common Tasks

### Update Knowledge Graph from lab_tutor-1
```bash
python3 utilities/run_knowledge_graph_update.py
```

### Clean Up Database
```bash
python3 utilities/cleanup_knowledge_graph.py
```

### View Statistics
```bash
python3 utilities/visualize_knowledge_graph.py
```

---

## Requirements

- Python 3.8+
- Neo4j running on `bolt://localhost:7687`
- Credentials: `neo4j` / `ktcd_password123`
- `neo4j` Python package: `pip install neo4j`

---

## Important Notes

1. **Backup First**: Always backup Neo4j database before running cleanup scripts
2. **Neo4j Required**: All scripts require Neo4j to be running
3. **lab_tutor-1**: Update scripts require lab_tutor-1 directory with extraction JSON files

---

## Related Documentation

- **Main README**: `../README.md`
- **Implementation Summary**: `../IMPLEMENTATION_SUMMARY.md`
- **Step-by-Step Guide**: `../STEP_BY_STEP_GUIDE.md`
- **Lab Documentation**: `../docs/LAB_FIXES_AND_UPDATES.md`

