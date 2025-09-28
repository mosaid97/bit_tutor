# services/knowledge_graph/models/student_knowledge_graph.py

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import importlib.util

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

class StudentKnowledgeGraph:
    """
    Manages the personalized knowledge graph for a single student.
    This graph now stores code submissions AND natural language questions,
    along with the system's evolving diagnosis of the student's mastery.
    
    Now integrates with Neo4j from lab_tutor to read and write knowledge graph data.
    
    The StudentKnowledgeGraph is the core memory system for the Educational Agent,
    serving as the single source of truth for all student data and learning activities.
    """
    def __init__(self, student_id, foundational_kg=None, q_matrix=None, hobbies=None):
        """
        Initialize a student knowledge graph.
        
        Args:
            student_id (str): The ID of the student
            foundational_kg (DiGraph, optional): The foundational knowledge graph
            q_matrix (DataFrame, optional): Q-matrix mapping exercises to knowledge components
            hobbies (list, optional): List of student hobbies for personalization
        """
        self.student_id = student_id
        self.graph = nx.DiGraph()
        self.q_matrix = q_matrix
        self.graph.add_node(student_id, type='student')
        
        # Store the student's hobbies for personalized themed content
        self.hobbies = hobbies if hobbies else []
        
        # Initialize history attributes
        self.history = []
        self.interaction_history = []
        
        # Initialize with empty KC nodes list
        self.kc_nodes = []
        
        # Initialize overall mastery
        self.overall_mastery = 0.1
        
        # Initialize cache - use dictionaries instead of None
        self._cache = {}
        self._cache['mastery_profile'] = {}
        self._cache['mastery_timestamp'] = None
        self._cache['centrality'] = {}
        self._cache['centrality_timestamp'] = None
        self._cache['recommendations'] = []
        self._cache['recommendations_timestamp'] = None
        self._cache_timeout = 300  # 5 minutes in seconds
        
        # Initialize history attribute
        self.history = []
        self.interaction_history = []
        
        # Initialize with empty KC nodes list
        self.kc_nodes = []
        
        # Try to connect to Neo4j
        self.neo4j_service = self._connect_to_neo4j()
        
        # Initialize knowledge components from either Neo4j or local graph
        if self.neo4j_service:
            self._init_from_neo4j()
        elif foundational_kg:
            self._init_from_local_kg(foundational_kg)
        else:
            # Create a default knowledge graph
            self._init_from_local_kg(self._create_default_kg())
            
        self.interaction_count = 0
        self.last_interaction_node = None
        
        # Store the overall mastery level for the student (average of all KCs)
        self.overall_mastery = 0.1  # Start with a low baseline
        self._update_overall_mastery()  # Calculate initial mastery
    
    def _connect_to_neo4j(self):
        """Attempt to connect to Neo4j database from lab_tutor"""
        # First check if Neo4jService is available
        if Neo4jService is None:
            print("Neo4jService is not available, falling back to local knowledge graph")
            return None
        
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
            print("Falling back to local knowledge graph")
            return None
        
        # Try to connect to Neo4j
        try:
            neo4j_service = Neo4jService()
            print(f"Connected to Neo4j for student {self.student_id}")
            return neo4j_service
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Falling back to local knowledge graph")
            return None
    
    def _init_from_neo4j(self):
        """Initialize knowledge components from Neo4j database"""
        if self.neo4j_service is None:
            print("Neo4j service not available, falling back to local knowledge graph")
            self._init_from_local_kg(self._create_default_kg())
            return
            
        try:
            # Get all CONCEPT nodes from Neo4j
            results = self.neo4j_service.query(
                "MATCH (c:CONCEPT) RETURN c.id as id, c.name as name"
            )
            
            self.kc_nodes = []
            for result in results:
                kc_id = result.get('id', f"neo4j_concept_{len(self.kc_nodes)}")
                if kc_id is None:
                    # Generate an ID if none exists
                    kc_id = f"neo4j_concept_{len(self.kc_nodes)}"
                    
                kc_name = result.get('name', 'Unknown Concept')
                self.graph.add_node(kc_id, name=kc_name, type='kc')
                self.graph.add_edge(self.student_id, kc_id, mastery=0.1)
                self.kc_nodes.append(kc_id)
            
            # Get relationships between concepts
            rel_results = self.neo4j_service.query(
                "MATCH (c1:CONCEPT)-[r]->(c2:CONCEPT) " +
                "RETURN c1.id as source, c2.id as target, type(r) as relationship"
            )
            
            for rel in rel_results:
                source = rel.get('source')
                target = rel.get('target')
                rel_type = rel.get('relationship')
                if source and target and source in self.kc_nodes and target in self.kc_nodes:
                    self.graph.add_edge(source, target, type=rel_type)
            
            # If no nodes were found, fall back to local knowledge graph
            if not self.kc_nodes:
                print("No CONCEPT nodes found in Neo4j, falling back to local knowledge graph")
                self._init_from_local_kg(self._create_default_kg())
            else:
                print(f"Initialized {len(self.kc_nodes)} knowledge components from Neo4j")
            
        except Exception as e:
            print(f"Error initializing from Neo4j: {e}")
            print("Falling back to local knowledge graph")
            self._init_from_local_kg(self._create_default_kg())
    
    def _init_from_local_kg(self, foundational_kg):
        """
        Initialize knowledge components from local NetworkX graph
        
        Args:
            foundational_kg (DiGraph): The foundational knowledge graph
        """
        try:
            # Safely extract KC nodes from foundational graph
            self.kc_nodes = []
            for node in foundational_kg.nodes():
                try:
                    if foundational_kg.nodes[node].get('type') == 'kc':
                        # Get name with fallback
                        name = foundational_kg.nodes[node].get('name', f'Concept {node}')
                        # Add KC to student graph
                        self.graph.add_node(node, name=name, type='kc')
                        # Initialize mastery edge with default value
                        self.graph.add_edge(self.student_id, node, mastery=0.1)
                        # Keep track of KC nodes
                        self.kc_nodes.append(node)
                except Exception as e:
                    print(f"Error adding KC node {node}: {e}")
                    # Continue with next node
                    continue
                    
            # Add relationships between KCs if they exist in foundational graph
            for source, target in foundational_kg.edges():
                if (source in self.kc_nodes and target in self.kc_nodes and 
                    source in self.graph.nodes and target in self.graph.nodes):
                    try:
                        # Copy edge attributes
                        edge_attrs = foundational_kg.edges[source, target]
                        self.graph.add_edge(source, target, **edge_attrs)
                    except Exception as e:
                        print(f"Error adding edge {source}->{target}: {e}")
                        
            print(f"Initialized {len(self.kc_nodes)} knowledge components for {self.student_id}")
            
            # Calculate initial overall mastery
            self._update_overall_mastery()
        except Exception as e:
            print(f"Error initializing from local KG: {e}")
            # Create empty KC nodes list if initialization fails
            if not hasattr(self, 'kc_nodes') or not self.kc_nodes:
                self.kc_nodes = []
    
    def _create_default_kg(self):
        """
        Create a default knowledge graph if Neo4j fails
        
        Returns:
            DiGraph: A default foundational knowledge graph
        """
        try:
            # Import build_cognitive_foundation from the same service
            from .cognitive_foundation import build_cognitive_foundation
            kg, _, _ = build_cognitive_foundation()
            return kg
        except Exception as e:
            print(f"Error creating default KG: {e}")
            # Return a minimal knowledge graph with basic KCs
            graph = nx.DiGraph()
            
            # Create some basic knowledge components
            basic_kcs = {
                'python_syntax': 'Python Syntax',
                'variables': 'Variables',
                'data_types': 'Data Types',
                'conditionals': 'Conditional Statements',
                'loops': 'Loops',
                'functions': 'Functions',
                'classes': 'Classes and Objects'
            }
            
            # Add nodes to graph
            for kc_id, name in basic_kcs.items():
                graph.add_node(kc_id, name=name, type='kc')
                
            # Add some basic relationships
            graph.add_edge('python_syntax', 'variables', type='prerequisite')
            graph.add_edge('variables', 'data_types', type='prerequisite')
            graph.add_edge('data_types', 'conditionals', type='prerequisite')
            graph.add_edge('conditionals', 'loops', type='prerequisite')
            graph.add_edge('loops', 'functions', type='prerequisite')
            graph.add_edge('functions', 'classes', type='prerequisite')
            
            return graph

    def add_interaction(self, interaction_data):
        """
        Adds a new interaction event to the graph. This can now be a
        'code_submission' or a 'question_asked' event.
        Also syncs with Neo4j if available.
        """
        self.interaction_count += 1
        interaction_id = f"int_{self.interaction_count}"
        interaction_type = interaction_data.get('type', 'code_submission')

        # Add a new node for this specific interaction event
        node_data = interaction_data.copy()
        node_data['type'] = 'interaction'
        node_data['interaction_type'] = interaction_type
        self.graph.add_node(interaction_id, **node_data)

        # Connect the student to this interaction
        self.graph.add_edge(self.student_id, interaction_id, type='performed')

        # Create a sequential link between interactions for the KT model
        if self.last_interaction_node:
            self.graph.add_edge(self.last_interaction_node, interaction_id, type='sequential')
        self.last_interaction_node = interaction_id

        # Sync with Neo4j if available
        if self.neo4j_service:
            try:
                # Create a query to add the interaction to Neo4j
                if interaction_type == 'code_submission':
                    query = """
                    CREATE (i:INTERACTION {
                        id: $id,
                        type: 'code_submission',
                        exercise_id: $exercise_id,
                        is_correct: $is_correct,
                        student_id: $student_id,
                        timestamp: datetime($timestamp)
                    })
                    WITH i
                    MATCH (s:STUDENT {id: $student_id})
                    MERGE (s)-[:PERFORMED]->(i)
                    """
                    params = {
                        'id': interaction_id,
                        'exercise_id': interaction_data.get('exercise_id', 'unknown'),
                        'is_correct': interaction_data.get('is_correct', False),
                        'student_id': self.student_id,
                        'timestamp': interaction_data.get('timestamp', '').isoformat()
                    }

                    # Execute the query
                    self.neo4j_service.query(query, params)
                    print(f"Synced '{interaction_type}' event to Neo4j")
            except Exception as e:
                print(f"Warning: Failed to sync interaction with Neo4j: {e}")

        print(f"Logged '{interaction_type}' event '{interaction_id}' to {self.student_id}'s graph.")

    def update_mastery_profile(self, mastery_dict):
        """
        Updates the mastery weights on the student->KC edges from the CD model.
        Also syncs with Neo4j if available.

        Args:
            mastery_dict (dict): Dictionary mapping KC IDs to mastery values
        """
        try:
            for kc_id, mastery_level in mastery_dict.items():
                if kc_id in self.kc_nodes:
                    # Ensure the edge exists
                    if not self.graph.has_edge(self.student_id, kc_id):
                        self.graph.add_edge(self.student_id, kc_id, mastery=round(mastery_level, 2))
                    else:
                        # Update existing edge
                        self.graph[self.student_id][kc_id]['mastery'] = round(mastery_level, 2)

                    # Sync with Neo4j if available
                    if self.neo4j_service:
                        try:
                            # Create a query to update mastery in Neo4j
                            query = """
                            MATCH (s:STUDENT {id: $student_id})-[r:KNOWS]->(c:CONCEPT {id: $concept_id})
                            SET r.mastery = $mastery
                            """
                            params = {
                                'student_id': self.student_id,
                                'concept_id': kc_id,
                                'mastery': round(mastery_level, 2)
                            }

                            # Execute the query
                            self.neo4j_service.query(query, params)
                        except Exception as e:
                            print(f"Warning: Failed to sync mastery with Neo4j: {e}")

            # Clear the mastery profile cache
            self._cache['mastery_profile'] = {}
            self._cache['mastery_timestamp'] = None

            # Update the overall mastery level
            self._update_overall_mastery()

            # Save the updated graph to file
            self.save_to_file()

        except Exception as e:
            print(f"Error updating mastery profile: {e}")

    def save_to_file(self, filepath=None):
        """
        Save the student knowledge graph to a file for persistence.

        Args:
            filepath (str, optional): Path to save the file. If None, uses default location.
        """
        try:
            # Generate default filepath if not provided
            if filepath is None:
                import os
                data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
                os.makedirs(data_dir, exist_ok=True)
                filepath = os.path.join(data_dir, f'student_{self.student_id}.pkl')

            # Save the graph using NetworkX functionality
            import pickle
            with open(filepath, 'wb') as f:
                # Create a dictionary with all the necessary data
                data = {
                    'student_id': self.student_id,
                    'graph': self.graph,
                    'kc_nodes': self.kc_nodes,
                    'overall_mastery': self.overall_mastery,
                    'history': self.history,
                    'interaction_history': self.interaction_history,
                    'q_matrix': self.q_matrix,
                    'hobbies': self.hobbies
                }
                pickle.dump(data, f)

            print(f"Saved student knowledge graph to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving student graph to file: {e}")
            return None

    @classmethod
    def load_from_file(cls, filepath):
        """
        Load a student knowledge graph from a file.

        Args:
            filepath (str): Path to the saved file

        Returns:
            StudentKnowledgeGraph: Loaded student knowledge graph
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Create a new instance
            student_kg = cls(
                student_id=data['student_id'],
                q_matrix=data.get('q_matrix'),
                hobbies=data.get('hobbies', [])
            )

            # Restore the graph and other attributes
            student_kg.graph = data['graph']
            student_kg.kc_nodes = data.get('kc_nodes', [])
            student_kg.overall_mastery = data.get('overall_mastery', 0.1)
            student_kg.history = data.get('history', [])
            student_kg.interaction_history = data.get('interaction_history', [])

            print(f"Loaded student knowledge graph from {filepath}")
            return student_kg
        except Exception as e:
            print(f"Error loading student graph from file: {e}")
            return None

    def _update_overall_mastery(self):
        """
        Calculate and update the overall mastery level based on all KCs.
        Uses node centrality to weight the importance of different KCs.
        """
        try:
            if not hasattr(self, 'kc_nodes') or not self.kc_nodes:
                self.overall_mastery = 0.1
                return

            # Check if we have enough nodes for centrality calculation
            if len(self.kc_nodes) < 2:
                # Simple average if not enough nodes
                mastery_values = [
                    self.graph.edges[self.student_id, kc].get('mastery', 0.1)
                    for kc in self.kc_nodes
                    if self.graph.has_edge(self.student_id, kc)
                ]
                if mastery_values:
                    self.overall_mastery = sum(mastery_values) / len(mastery_values)
                else:
                    self.overall_mastery = 0.1
                return

            # Create a subgraph of just the KCs to calculate centrality
            kc_subgraph = self.graph.subgraph(self.kc_nodes)

            try:
                # Calculate centrality - try degree centrality first as it's most reliable
                centrality = nx.degree_centrality(kc_subgraph)

                # Normalize centrality
                total_centrality = sum(centrality.values())
                if total_centrality == 0:  # If all nodes have the same centrality
                    centrality = {node: 1.0/len(self.kc_nodes) for node in self.kc_nodes}
                else:
                    centrality = {node: value/total_centrality for node, value in centrality.items()}
            except Exception:
                # Fallback to equal weights if centrality calculation fails
                centrality = {node: 1.0/len(self.kc_nodes) for node in self.kc_nodes}

            # Calculate weighted mastery
            weighted_mastery = 0.0
            total_weight = 0.0

            for kc in self.kc_nodes:
                if self.graph.has_edge(self.student_id, kc):
                    mastery = self.graph.edges[self.student_id, kc].get('mastery', 0.1)
                    weight = centrality.get(kc, 1.0/len(self.kc_nodes))
                    weighted_mastery += mastery * weight
                    total_weight += weight

            # Update overall mastery
            if total_weight > 0:
                self.overall_mastery = weighted_mastery / total_weight
            else:
                self.overall_mastery = 0.1

            # Cap mastery between 0 and 1
            self.overall_mastery = max(0.0, min(1.0, self.overall_mastery))

        except Exception as e:
            print(f"Error updating overall mastery: {e}")
            self.overall_mastery = 0.1

    def get_mastery_profile(self):
        """
        Returns the current mastery profile of the student.
        This is a dictionary mapping KC IDs to mastery levels.
        Uses caching to improve performance for frequent calls.

        Returns:
            dict: Dictionary mapping KC IDs to mastery values
        """
        # Check if we have a valid cached profile
        if (self._cache.get('mastery_timestamp') is not None and
            (datetime.now() - self._cache['mastery_timestamp']).total_seconds() < self._cache_timeout):
            return self._cache['mastery_profile']

        try:
            if not hasattr(self, 'kc_nodes') or not self.kc_nodes:
                self._cache['mastery_profile'] = {}
                self._cache['mastery_timestamp'] = datetime.now()
                return {}

            mastery_profile = {}
            for kc in self.kc_nodes:
                if self.graph.has_edge(self.student_id, kc):
                    mastery_profile[kc] = self.graph.edges[self.student_id, kc].get('mastery', 0.1)
                else:
                    # Default mastery level for KCs without an edge
                    mastery_profile[kc] = 0.1

            # Update cache
            self._cache['mastery_profile'] = mastery_profile
            self._cache['mastery_timestamp'] = datetime.now()

            return mastery_profile

        except Exception as e:
            print(f"Error in get_mastery_profile: {e}")
            # Return empty dictionary on error
            return {}

    def set_mastery_profile(self, mastery_profile):
        """
        Sets the mastery profile for the student.

        Args:
            mastery_profile (dict): A dictionary mapping KC IDs to mastery levels.
        """
        try:
            for kc_id, mastery_level in mastery_profile.items():
                # Make sure the KC node exists
                if kc_id not in self.graph.nodes:
                    self.graph.add_node(kc_id, name=kc_id, type='kc')
                    if not hasattr(self, 'kc_nodes'):
                        self.kc_nodes = []
                    if kc_id not in self.kc_nodes:
                        self.kc_nodes.append(kc_id)

                # Update or create the edge
                if self.graph.has_edge(self.student_id, kc_id):
                    self.graph[self.student_id][kc_id]['mastery'] = mastery_level
                else:
                    self.graph.add_edge(self.student_id, kc_id, mastery=mastery_level)

            # Update overall mastery
            self._update_overall_mastery()

            return True
        except Exception as e:
            print(f"Error in set_mastery_profile: {e}")
            return False
