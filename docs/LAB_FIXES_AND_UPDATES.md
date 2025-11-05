# Lab Fixes and Knowledge Graph Updates

## Overview

This document describes the fixes implemented for the lab system and the knowledge graph update from lab_tutor-1.

## Issues Fixed

### 1. Lab Jupyter Notebook Link Not Working

**Problem:**
- The "Open Lab Environment" button in the topic learning page was linking to an external URL (`{{ lab.url }}`)
- Students could not access the interactive Jupyter-style notebook
- The lab_id was not being fetched from the database

**Solution:**
- Updated `templates/student/topic_learning_tabbed.html` to link to the internal Jupyter notebook route
- Changed the button to redirect to `/student/<student_id>/learn/topics/<topic_name>/labs/<lab_id>`
- Updated the route query in `routes/student_learning_routes.py` to fetch `lab_id`
- Added fallback to generate `lab_id` if not present in database

**Files Modified:**
- `templates/student/topic_learning_tabbed.html` (lines 232-272)
- `routes/student_learning_routes.py` (lines 225-238)

**Changes:**
```python
# Before (external link)
<a href="{{ lab.url }}" target="_blank">Open Lab Environment</a>

# After (internal Jupyter notebook)
<a href="/student/{{ student_id }}/learn/topics/{{ topic.name }}/labs/{{ lab.lab_id }}">
    Open Interactive Notebook
</a>
```

### 2. Knowledge Graph Update from lab_tutor-1

**Problem:**
- The knowledge graph needed to be updated with new topics and concepts from lab_tutor-1
- lab_tutor-1 contains extracted JSON files with comprehensive topic information
- The existing knowledge graph was outdated

**Solution:**
Created three utility scripts to handle the update:

#### Script 1: `utilities/update_from_lab_tutor_1.py`
- Reads all extraction JSON files from `lab_tutor-1/knowledge_graph_builder/batch_output/`
- Extracts topics, concepts, theories, and keywords
- Updates Neo4j knowledge graph with new data
- Links topics to the "Big Data Analysis" class
- Creates proper relationships: Class→Topic→Theory→Concept

**Features:**
- Automatic topic discovery from JSON files
- Preserves existing data while adding new content
- Creates proper Neo4j relationships
- Provides statistics after update

#### Script 2: `utilities/create_comprehensive_labs.py`
- Creates comprehensive Jupyter-style labs for each topic
- Each lab covers ALL concepts under the topic
- Generates interactive coding cells with:
  - Setup and imports
  - One cell per concept
  - Integration exercise
  - Visualization and analysis
- Stores lab data in Neo4j with proper relationships

**Lab Structure:**
```json
{
  "lab_id": "lab_topic_name",
  "title": "Comprehensive Lab: Topic Name",
  "objective": "Master all concepts through hands-on coding",
  "difficulty": "intermediate",
  "estimated_time": 150,  // 15 min per concept
  "cells": [
    {
      "title": "Setup and Imports",
      "instructions": "Import necessary libraries",
      "code": "import numpy as np...",
      "hint": "Make sure all libraries are installed"
    },
    // ... cells for each concept
  ]
}
```

#### Script 3: `utilities/run_knowledge_graph_update.py`
- Master script that runs both updates in sequence
- Verifies the update was successful
- Provides statistics and next steps

## Knowledge Graph Structure

After the update, the knowledge graph has the following structure:

```
Class (Big Data Analysis)
  ├─[INCLUDES]→ Topic 1
  │   ├─[HAS_THEORY]→ Theory 1
  │   │   └─[CONSISTS_OF]→ Concept 1.1, 1.2, 1.3...
  │   ├─[INCLUDES_CONCEPT]→ Concept 1.1, 1.2, 1.3...
  │   └─[PRACTICES]←─ Lab 1
  │       └─[APPLIES]→ Concept 1.1, 1.2, 1.3...
  ├─[INCLUDES]→ Topic 2
  │   └─ ...
  └─ ...
```

## How to Run the Update

### Option 1: Run Complete Update (Recommended)

```bash
cd /Users/mohasani/KTCD_Aug
python3 utilities/run_knowledge_graph_update.py
```

This will:
1. Update topics and concepts from lab_tutor-1
2. Create comprehensive labs for all topics
3. Verify the update was successful

### Option 2: Run Individual Scripts

```bash
# Step 1: Update topics and concepts
python3 utilities/update_from_lab_tutor_1.py

# Step 2: Create comprehensive labs
python3 utilities/create_comprehensive_labs.py
```

### Option 3: Verify Only

```python
from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager

graph_manager = DynamicGraphManager()

# Check topics
topics_query = """
MATCH (c:Class {name: 'Big Data Analysis'})-[:INCLUDES]->(t:Topic)
RETURN count(t) as count
"""
result = graph_manager.neo4j.graph.query(topics_query)
print(f"Topics: {result[0]['count']}")
```

## Expected Results

After running the update, you should see:

- **Topics**: 20+ topics from lab_tutor-1
- **Theories**: One theory per topic
- **Concepts**: 100+ concepts across all topics
- **Labs**: One comprehensive lab per topic

## Testing the Fixes

### Test Lab Notebook Access

1. Start the Flask application:
   ```bash
   python3 nexus_app.py
   ```

2. Login as a student (e.g., Roma)

3. Navigate to a topic learning page

4. Click the "Lab" tab

5. Click "Open Interactive Notebook"

6. Verify:
   - Jupyter-style notebook loads
   - CodeMirror editor is functional
   - Theory section shows concepts
   - Cells can be executed
   - Hints are available

### Test Knowledge Graph Update

1. Check Neo4j Browser at http://localhost:7474

2. Run this query:
   ```cypher
   MATCH (c:Class {name: 'Big Data Analysis'})-[:INCLUDES]->(t:Topic)
   OPTIONAL MATCH (t)-[:HAS_THEORY]->(th:Theory)
   OPTIONAL MATCH (th)-[:CONSISTS_OF]->(concept:Concept)
   OPTIONAL MATCH (l:Lab)-[:PRACTICES]->(t)
   RETURN t.name as topic, 
          count(DISTINCT concept) as concepts,
          count(DISTINCT l) as labs
   ORDER BY topic
   ```

3. Verify each topic has:
   - At least 3 concepts
   - Exactly 1 comprehensive lab

## Troubleshooting

### Lab Notebook Not Loading

**Issue**: 404 error when clicking "Open Interactive Notebook"

**Solution**:
- Check that `lab_id` exists in the database
- Verify the route is registered: `/student/<student_id>/learn/topics/<topic_name>/labs/<lab_id>`
- Check Flask logs for errors

### Knowledge Graph Update Fails

**Issue**: Script cannot connect to Neo4j

**Solution**:
- Verify Neo4j is running: `docker ps`
- Check Neo4j credentials in `.env` file
- Test connection: `python3 -c "from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager; DynamicGraphManager()"`

### No Topics Found After Update

**Issue**: Update script reports "No topics found"

**Solution**:
- Verify lab_tutor-1 directory exists
- Check that JSON files are in `lab_tutor-1/knowledge_graph_builder/batch_output/`
- Ensure "Big Data Analysis" class exists in Neo4j

## Next Steps

After completing the fixes and updates:

1. **Restart Flask Application**: Ensure all changes are loaded
2. **Test Student Flow**: Login as a student and test the complete learning flow
3. **Verify Labs**: Check that all topics have working labs
4. **Monitor Logs**: Watch for any errors during lab execution
5. **Update Documentation**: Document any additional changes or customizations

## Additional Resources

- **Lab Notebook Template**: `templates/student/lab_notebook.html`
- **Lab Routes**: `routes/student_learning_routes.py` (lines 657-759)
- **Graph Manager**: `services/knowledge_graph/services/dynamic_graph_manager.py`
- **Neo4j Browser**: http://localhost:7474

## Summary

✅ **Lab Jupyter notebook link fixed** - Students can now access interactive notebooks
✅ **Knowledge graph update scripts created** - Easy update from lab_tutor-1
✅ **Comprehensive labs generated** - One lab per topic covering all concepts
✅ **Documentation complete** - Clear instructions for running updates

The system is now ready for students to use the interactive Jupyter-style labs!

