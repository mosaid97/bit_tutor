# services/knowledge_graph/models/cognitive_foundation.py

import pandas as pd
import networkx as nx
import os
import sys
import importlib.util
from datetime import datetime

# Try to import Neo4jService using importlib
Neo4jService = None
try:
    # First try direct import
    try:
        from services.neo4j_service import Neo4jService
        print("Successfully imported Neo4jService from services.neo4j_service")
    except ImportError:
        # Try with full path using importlib
        lab_tutor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'lab_tutor')
        knowledge_graph_builder_path = os.path.join(lab_tutor_path, 'knowledge_graph_builder')
        services_path = os.path.join(knowledge_graph_builder_path, 'services')
        neo4j_service_path = os.path.join(services_path, 'neo4j_service.py')
        if os.path.exists(neo4j_service_path):
            spec = importlib.util.spec_from_file_location("neo4j_service", neo4j_service_path)
            neo4j_service_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(neo4j_service_module)
            Neo4jService = neo4j_service_module.Neo4jService
            print(f"Successfully imported Neo4jService using importlib from {neo4j_service_path}")
        else:
            print(f"Neo4j service file not found at {neo4j_service_path}")
            Neo4jService = None
except Exception as e:
    print(f"Warning: Could not import Neo4jService: {e}")
    print("Falling back to local knowledge graph implementation")
    Neo4jService = None

def build_cognitive_foundation():
    """
    Builds the cognitive foundation by either loading from Neo4j (if available)
    or using the static curriculum graph and Q-Matrix.
    """
    # First check if Neo4jService is available
    if Neo4jService is None:
        print("Neo4jService is not available, using local cognitive foundation")
        return _build_default_foundation()
    
    # Check if Neo4j server is accessible
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)  # Set timeout to 2 seconds
        s.connect(('localhost', 7687))
        s.close()
    except (socket.error, socket.timeout):
        print("Neo4j server is not accessible at localhost:7687")
        print("Make sure the Neo4j Docker container is running")
        print("Falling back to local cognitive foundation")
        return _build_default_foundation()
    
    # Try to connect to Neo4j and load data
    try:
        # Create Neo4j service
        neo4j_service = Neo4jService()
        print("Successfully connected to Neo4j for cognitive foundation")
        
        # Get concepts from Neo4j
        concept_results = neo4j_service.query(
            "MATCH (c:CONCEPT) RETURN c.id as id, c.name as name"
        )
        
        if not concept_results:
            print("No CONCEPT nodes found in Neo4j, falling back to local cognitive foundation")
            return _build_default_foundation()
            
        knowledge_components = {}
        knowledge_graph = nx.DiGraph()
        
        for concept in concept_results:
            kc_id = concept.get('id')
            if kc_id is None:
                # Generate an ID if none exists
                kc_id = f"neo4j_concept_{len(knowledge_components)}"
                
            kc_name = concept.get('name', 'Unknown Concept')
            knowledge_components[kc_id] = kc_name
            knowledge_graph.add_node(kc_id, name=kc_name, type='kc')
        
        # Get relationships between concepts
        rel_results = neo4j_service.query(
            "MATCH (c1:CONCEPT)-[r]->(c2:CONCEPT) " +
            "RETURN c1.id as source, c2.id as target, type(r) as relationship"
        )
        
        for rel in rel_results:
            source = rel.get('source')
            target = rel.get('target')
            if source and target and source in knowledge_components and target in knowledge_components:
                knowledge_graph.add_edge(source, target)
        
        print(f"Loaded {len(knowledge_components)} concepts from Neo4j")
        
        # Create a Q-matrix based on concepts
        q_matrix_data = {}
        concept_ids = list(knowledge_components.keys())
        
        # Create a Q-matrix with some sample exercises
        # In a real system, this would be loaded from a database
        for i in range(min(5, len(concept_ids))):
            ex_id = f"ex_{i+1}"
            q_matrix_data[ex_id] = [0] * len(concept_ids)
            
            # Each exercise tests 1-3 concepts
            num_concepts = min(3, len(concept_ids))
            for j in range(num_concepts):
                idx = (i + j) % len(concept_ids)
                q_matrix_data[ex_id][idx] = 1
        
        q_matrix = pd.DataFrame.from_dict(q_matrix_data, orient='index', columns=concept_ids)
        
        return knowledge_graph, q_matrix, knowledge_components
        
    except Exception as e:
        print(f"Error in Neo4j operations: {e}")
        print("Falling back to local cognitive foundation")
        return _build_default_foundation()

def _build_default_foundation():
    """Build the default foundation when Neo4j is not available"""
    knowledge_components = {
        "KC1": "Variables", "KC2": "Data Types", "KC3": "Operators",
        "KC4": "Printing", "KC5": "Conditionals", "KC6": "For Loops",
        "KC7": "Lists", "KC8": "Functions"
    }
    prerequisites = [("KC1", "KC3"), ("KC2", "KC3"), ("KC1", "KC5"), ("KC7", "KC6")]
    
    knowledge_graph = nx.DiGraph()
    for kc_id, kc_name in knowledge_components.items():
        knowledge_graph.add_node(kc_id, name=kc_name, type='kc')
    knowledge_graph.add_edges_from(prerequisites)
    
    q_matrix_data = {
        "ex1": [1, 1, 1, 0, 0, 0, 0, 0], # Variables, Data Types, Operators
        "ex2": [0, 0, 0, 1, 0, 0, 0, 0], # Printing
        "ex3": [1, 1, 0, 0, 1, 0, 0, 0], # Variables, Data Types, Conditionals
        "ex4": [1, 0, 0, 0, 0, 1, 1, 0], # Variables, For Loops, Lists
        "ex5": [1, 1, 0, 0, 0, 0, 0, 1]  # Variables, Data Types, Functions
    }
    q_matrix = pd.DataFrame.from_dict(q_matrix_data, orient='index', columns=list(knowledge_components.keys()))
    
    return knowledge_graph, q_matrix, knowledge_components

def run_educational_agent(student_id, interaction):
    """
    Main educational agent function that implements the Perceive -> Reason -> Plan -> Act -> Learn cycle.
    This is the core logic for the autonomous AI Educational Agent.
    
    Args:
        student_id (str): ID of the student
        interaction (dict): New interaction data from the student
        
    Returns:
        dict: JSON response with the agent's decision and explanation
    """
    try:
        # Import StudentKnowledgeGraph from the same service
        from .student_knowledge_graph import StudentKnowledgeGraph
        
        # Load cognitive foundation
        foundational_kg, q_matrix, kcs = build_cognitive_foundation()
        
        # Initialize or load student knowledge graph
        try:
            # Check if we have a saved graph for this student
            import pickle
            import os
            
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
            student_file = os.path.join(data_dir, f'{student_id}.pkl')
            
            if os.path.exists(student_file):
                with open(student_file, 'rb') as f:
                    student_graph = pickle.load(f)
                print(f"Loaded existing knowledge graph for {student_id}")
            else:
                # Create a new graph for this student
                student_graph = StudentKnowledgeGraph(student_id, foundational_kg, q_matrix)
                print(f"Created new knowledge graph for {student_id}")
        except Exception as e:
            print(f"Error loading student graph: {e}")
            # Create a new graph if loading fails
            student_graph = StudentKnowledgeGraph(student_id, foundational_kg, q_matrix)
        
        # PERCEIVE: Add the new interaction to the graph
        student_graph.add_interaction(interaction)
        print(f"Added new interaction to {student_id}'s graph")
        
        # DIAGNOSE: Update the knowledge state using the GNN model
        updated_mastery = student_graph.diagnose_knowledge()
        print(f"Updated mastery profile for {student_id}")
        
        # REASON & STRATEGIZE: Analyze the updated mastery profile
        # Find the weakest knowledge component(s)
        weakest_kcs = sorted(
            [(kc, student_graph.graph.edges[student_id, kc]['mastery']) 
             for kc in student_graph.kc_nodes
             if student_graph.graph.has_edge(student_id, kc)],
            key=lambda x: x[1]
        )
        
        # Extract information from the interaction
        interaction_type = interaction.get('type', '')
        
        # Formulate reasoning log
        reasoning_log = f"Student {student_id} has updated mastery profile.\n"
        
        if weakest_kcs:
            weakest_kc, mastery = weakest_kcs[0]
            kc_name = student_graph.graph.nodes[weakest_kc].get('name', 'Unknown')
            reasoning_log += f"Weakest KC: {kc_name} ({weakest_kc}) with mastery {mastery:.2f}\n"
        
        if interaction_type == 'code_submission':
            exercise_id = interaction.get('exercise_id', '')
            is_correct = interaction.get('is_correct', False)
            reasoning_log += f"Last interaction: code submission for {exercise_id} (correct: {is_correct})\n"
            
            # Add specific reasoning for code submissions
            if is_correct:
                reasoning_log += "Student successfully completed the exercise. Consider increasing difficulty.\n"
            else:
                reasoning_log += "Student struggled with the exercise. Consider providing targeted help.\n"
                
        elif interaction_type == 'question_asked':
            question = interaction.get('question_text', '')
            reasoning_log += f"Last interaction: question asked: '{question[:50]}...'\n"
            
            # Analyze question to determine next steps
            reasoning_log += "Question indicates student is seeking clarification. Providing explanation is appropriate.\n"
        
        # PLAN & SELECT ACTION: Determine the best next step
        # Simple decision logic: balance between exercises and explanations
        if interaction_type == 'question_asked' or (interaction_type == 'code_submission' and not interaction.get('is_correct', False)):
            # If student asked a question or failed an exercise, provide explanation
            if weakest_kcs:
                action_type = 'generate_content'
                target_kc = weakest_kcs[0][0]
                content_type = 'explanation'
                difficulty = 'medium'
                
                chosen_action = {
                    'type': action_type,
                    'target_kc': target_kc,
                    'content_type': content_type,
                    'difficulty': difficulty
                }
                reasoning_log += f"Decision: Generate {content_type} for {target_kc} to address misunderstanding.\n"
                
                # Generate the content
                content = student_graph.generate_content(content_type, target_kc, difficulty)
                chosen_action['content'] = content
            else:
                # Fallback to recommending an exercise
                action_type = 'recommend_exercise'
                exercise_id = list(q_matrix.index)[0]  # Default to first exercise
                
                chosen_action = {
                    'type': action_type,
                    'exercise_id': exercise_id
                }
                reasoning_log += f"Decision: Recommend exercise {exercise_id} as fallback.\n"
                
                # Get exercise details
                exercise_details = student_graph.recommend_exercise(exercise_id)
                chosen_action['details'] = exercise_details
        else:
            # Otherwise recommend a new exercise
            action_type = 'recommend_exercise'
            
            # Find exercises that target the weakest KC
            if weakest_kcs:
                target_kc = weakest_kcs[0][0]
                suitable_exercises = []
                
                for exercise_id in q_matrix.index:
                    kc_idx = list(q_matrix.columns).index(target_kc) if target_kc in q_matrix.columns else -1
                    if kc_idx >= 0 and q_matrix.loc[exercise_id].iloc[kc_idx] > 0:
                        suitable_exercises.append(exercise_id)
                
                if suitable_exercises:
                    exercise_id = suitable_exercises[0]
                else:
                    # Fallback to first exercise
                    exercise_id = list(q_matrix.index)[0]
            else:
                # Fallback to first exercise
                exercise_id = list(q_matrix.index)[0]
            
            chosen_action = {
                'type': action_type,
                'exercise_id': exercise_id
            }
            reasoning_log += f"Decision: Recommend exercise {exercise_id} targeting weakest KC.\n"
            
            # Get exercise details
            exercise_details = student_graph.recommend_exercise(exercise_id)
            chosen_action['details'] = exercise_details
        
        # ACT & EXPLAIN: Generate explanation for the decision
        explanation = student_graph.explain_decision(reasoning_log, chosen_action)
        
        # Save updated student graph
        try:
            import pickle
            import os
            
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
            os.makedirs(data_dir, exist_ok=True)
            student_file = os.path.join(data_dir, f'{student_id}.pkl')
            
            with open(student_file, 'wb') as f:
                pickle.dump(student_graph, f)
            print(f"Saved updated knowledge graph for {student_id}")
        except Exception as e:
            print(f"Error saving student graph: {e}")
        
        # Format the final output as JSON
        output = {
            'student_id': student_id,
            'timestamp': datetime.now().isoformat(),
            'chosen_action': chosen_action,
            'explanation': explanation,
            'reasoning_log': reasoning_log,
            'mastery_profile': {
                kc: student_graph.graph.edges[student_id, kc]['mastery']
                for kc in student_graph.kc_nodes
                if student_graph.graph.has_edge(student_id, kc)
            }
        }
        
        return output
        
    except Exception as e:
        print(f"Error in educational agent: {e}")
        # Return a simple error response
        return {
            'error': str(e),
            'student_id': student_id,
            'timestamp': datetime.now().isoformat(),
            'chosen_action': {
                'type': 'error',
                'message': 'An error occurred while processing the interaction'
            }
        }
