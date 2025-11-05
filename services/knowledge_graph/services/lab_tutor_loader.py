# services/knowledge_graph/services/lab_tutor_loader.py

"""
Lab Tutor Content Loader

This service loads topics, concepts, and theories from the static lab_tutor knowledge base.
It provides a read-only interface to the containerized content.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class LabTutorLoader:
    """
    Loads and provides access to the static lab_tutor knowledge base.
    
    This class reads the complete_neo4j_export_no_embeddings.json file
    and provides methods to query topics, concepts, and theories.
    """
    
    def __init__(self, lab_tutor_path: Optional[str] = None):
        """
        Initialize the Lab Tutor Loader.
        
        Args:
            lab_tutor_path: Path to lab_tutor directory. If None, uses default path.
        """
        if lab_tutor_path is None:
            # Default path relative to this file
            base_path = Path(__file__).parent.parent.parent.parent
            lab_tutor_path = base_path / "lab_tutor" / "knowledge_graph_builder"
        
        self.lab_tutor_path = Path(lab_tutor_path)
        self.data_file = self.lab_tutor_path / "complete_neo4j_export_no_embeddings.json"
        
        # Load data
        self.data = self._load_data()
        
        # Index data for quick access
        self.topics_index = self._index_topics()
        self.theories_index = self._index_theories()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load the JSON data from lab_tutor."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded lab_tutor data: {len(data.get('topics', []))} topics, {len(data.get('theories', []))} theories")
            return data
        except FileNotFoundError:
            print(f"⚠️  Warning: Lab tutor data file not found at {self.data_file}")
            return {"topics": [], "theories": []}
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Error parsing lab_tutor JSON: {e}")
            return {"topics": [], "theories": []}
    
    def _index_topics(self) -> Dict[str, Dict[str, Any]]:
        """Create an index of topics by name."""
        index = {}
        for topic in self.data.get('topics', []):
            topic_name = topic.get('name', '')
            if topic_name:
                index[topic_name] = topic
        return index
    
    def _index_theories(self) -> Dict[str, Dict[str, Any]]:
        """Create an index of theories by ID."""
        index = {}
        for theory in self.data.get('theories', []):
            theory_id = theory.get('id', '')
            if theory_id:
                index[theory_id] = theory
        return index
    
    def get_all_topics(self) -> List[Dict[str, Any]]:
        """
        Get all available topics.
        
        Returns:
            List of topic dictionaries
        """
        return self.data.get('topics', [])
    
    def get_topic_by_name(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific topic by name.
        
        Args:
            topic_name: Name of the topic
            
        Returns:
            Topic dictionary or None if not found
        """
        return self.topics_index.get(topic_name)
    
    def get_theories_for_topic(self, topic_name: str) -> List[Dict[str, Any]]:
        """
        Get all theories related to a topic.
        
        Args:
            topic_name: Name of the topic
            
        Returns:
            List of theory dictionaries
        """
        theories = []
        for theory in self.data.get('theories', []):
            # Check if theory is related to this topic
            # This is a simple keyword match - can be enhanced
            theory_text = theory.get('original_text', '').lower()
            compressed_text = theory.get('compressed_text', '').lower()
            keywords = [k.lower() for k in theory.get('keywords', [])]
            
            topic_lower = topic_name.lower()
            
            if (topic_lower in theory_text or 
                topic_lower in compressed_text or 
                any(topic_lower in keyword for keyword in keywords)):
                theories.append(theory)
        
        return theories
    
    def get_theory_by_id(self, theory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific theory by ID.
        
        Args:
            theory_id: ID of the theory
            
        Returns:
            Theory dictionary or None if not found
        """
        return self.theories_index.get(theory_id)
    
    def get_concepts_from_theories(self, theories: List[Dict[str, Any]]) -> List[str]:
        """
        Extract unique concepts from a list of theories.
        
        Args:
            theories: List of theory dictionaries
            
        Returns:
            List of unique concept names (keywords)
        """
        concepts = set()
        for theory in theories:
            keywords = theory.get('keywords', [])
            concepts.update(keywords)
        
        return sorted(list(concepts))
    
    def get_topic_summary(self, topic_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a topic including theories and concepts.
        
        Args:
            topic_name: Name of the topic
            
        Returns:
            Dictionary with topic information, theories, and concepts
        """
        topic = self.get_topic_by_name(topic_name)
        if not topic:
            return {
                "name": topic_name,
                "found": False,
                "theories": [],
                "concepts": []
            }
        
        theories = self.get_theories_for_topic(topic_name)
        concepts = self.get_concepts_from_theories(theories)
        
        return {
            "name": topic_name,
            "found": True,
            "theories": theories,
            "concepts": concepts,
            "theory_count": len(theories),
            "concept_count": len(concepts)
        }
    
    def search_topics(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for topics by keyword.
        
        Args:
            query: Search query
            
        Returns:
            List of matching topics
        """
        query_lower = query.lower()
        matching_topics = []
        
        for topic in self.data.get('topics', []):
            topic_name = topic.get('name', '').lower()
            if query_lower in topic_name:
                matching_topics.append(topic)
        
        return matching_topics
    
    def get_all_concepts(self) -> List[str]:
        """
        Get all unique concepts across all theories.
        
        Returns:
            List of unique concept names
        """
        all_theories = self.data.get('theories', [])
        return self.get_concepts_from_theories(all_theories)
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the lab_tutor content.
        
        Returns:
            Dictionary with counts of topics, theories, and concepts
        """
        all_concepts = self.get_all_concepts()
        
        return {
            "total_topics": len(self.data.get('topics', [])),
            "total_theories": len(self.data.get('theories', [])),
            "total_concepts": len(all_concepts)
        }


# Global instance
_lab_tutor_loader = None


def get_lab_tutor_loader() -> LabTutorLoader:
    """
    Get the global LabTutorLoader instance (singleton pattern).
    
    Returns:
        LabTutorLoader instance
    """
    global _lab_tutor_loader
    if _lab_tutor_loader is None:
        _lab_tutor_loader = LabTutorLoader()
    return _lab_tutor_loader

