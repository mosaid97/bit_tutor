# services/content_generation/services/lab_generator.py

"""
Lab Exercise Generator Service

Generates hands-on lab exercises for topics using LLM or templates.
Labs are practical exercises that students complete to apply concepts.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import random


class LabGenerator:
    """
    Generates hands-on lab exercises for topics.
    
    Features:
    - Generates 3-5 labs per topic
    - Practical, hands-on exercises
    - Step-by-step instructions
    - Covers concepts from the topic
    - Uses LLM or template-based generation
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the Lab Generator.
        
        Args:
            use_llm: Whether to use actual LLM (requires OpenAI API key)
        """
        self.use_llm = use_llm
        self.llm_client = None
        
        if use_llm:
            try:
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key)
                    print("✅ LLM Lab Generator initialized")
                else:
                    print("⚠️  No OpenAI API key found, using template-based lab generation")
                    self.use_llm = False
            except ImportError:
                print("⚠️  OpenAI package not installed, using template-based lab generation")
                self.use_llm = False
    
    def generate_labs_for_topic(self,
                                topic_name: str,
                                concepts: List[str],
                                theory_data: Optional[List[Dict[str, Any]]] = None,
                                num_labs: int = 4) -> List[Dict[str, Any]]:
        """
        Generate lab exercises for a topic.
        
        Args:
            topic_name: Name of the topic
            concepts: List of concepts in the topic
            theory_data: Optional theory data from lab_tutor
            num_labs: Number of labs to generate (default 4)
            
        Returns:
            List of lab exercise dictionaries
        """
        num_labs = max(3, min(6, num_labs))  # 3-6 labs
        
        if self.use_llm and self.llm_client:
            labs = self._generate_with_llm(topic_name, concepts, theory_data, num_labs)
        else:
            labs = self._generate_with_template(topic_name, concepts, theory_data, num_labs)
        
        return labs
    
    def _generate_with_llm(self,
                          topic_name: str,
                          concepts: List[str],
                          theory_data: Optional[List[Dict[str, Any]]],
                          num_labs: int) -> List[Dict[str, Any]]:
        """Generate labs using LLM."""
        labs = []
        
        # Create labs of different difficulty levels
        difficulties = ['beginner', 'intermediate', 'advanced']
        
        for i in range(num_labs):
            difficulty = difficulties[min(i, len(difficulties)-1)]
            
            prompt = f"""Create a hands-on lab exercise for "{topic_name}" at {difficulty} level.

Concepts to cover: {', '.join(concepts[:3])}

The lab should:
1. Have a clear objective
2. Include step-by-step instructions (5-8 steps)
3. Be practical and hands-on
4. Take 30-60 minutes to complete
5. Include expected outcomes

Format:
TITLE: [lab title]
OBJECTIVE: [what students will learn]
DIFFICULTY: {difficulty}
ESTIMATED_TIME: [time in minutes]
PREREQUISITES: [what students need to know]
INSTRUCTIONS:
1. [step 1]
2. [step 2]
...
EXPECTED_OUTCOME: [what students should achieve]
VERIFICATION: [how to verify completion]
"""
            
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert educational lab designer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                lab = self._parse_llm_lab(response.choices[0].message.content, topic_name, i+1)
                labs.append(lab)
                
            except Exception as e:
                print(f"Error generating LLM lab: {e}")
                # Fallback to template
                labs.append(self._generate_template_lab(topic_name, concepts, i+1, difficulty))
        
        return labs
    
    def _generate_with_template(self,
                                topic_name: str,
                                concepts: List[str],
                                theory_data: Optional[List[Dict[str, Any]]],
                                num_labs: int) -> List[Dict[str, Any]]:
        """Generate labs using templates."""
        labs = []
        difficulties = ['beginner', 'intermediate', 'advanced', 'expert']
        
        for i in range(num_labs):
            difficulty = difficulties[min(i, len(difficulties)-1)]
            lab = self._generate_template_lab(topic_name, concepts, i+1, difficulty)
            labs.append(lab)
        
        return labs
    
    def _generate_template_lab(self, topic: str, concepts: List[str], lab_num: int, difficulty: str) -> Dict[str, Any]:
        """Generate a template-based lab exercise."""
        
        # Select concepts for this lab
        lab_concepts = concepts[:min(3, len(concepts))]
        
        # Lab templates based on difficulty
        if difficulty == 'beginner':
            title = f"{topic} - Getting Started (Lab {lab_num})"
            objective = f"Learn the fundamentals of {topic} through hands-on practice"
            time = 30
            instructions = [
                f"Set up your development environment for {topic}",
                f"Create a simple example using {lab_concepts[0] if lab_concepts else 'basic concepts'}",
                f"Explore the basic features and functionality",
                f"Modify the example to understand how it works",
                f"Document your observations and findings"
            ]
        elif difficulty == 'intermediate':
            title = f"{topic} - Practical Application (Lab {lab_num})"
            objective = f"Apply {topic} concepts to solve real-world problems"
            time = 45
            instructions = [
                f"Review the requirements for a {topic} implementation",
                f"Design a solution using {', '.join(lab_concepts[:2])}",
                f"Implement the core functionality",
                f"Test your implementation with sample data",
                f"Optimize for performance and scalability",
                f"Document your design decisions"
            ]
        elif difficulty == 'advanced':
            title = f"{topic} - Advanced Techniques (Lab {lab_num})"
            objective = f"Master advanced {topic} techniques and best practices"
            time = 60
            instructions = [
                f"Analyze a complex {topic} scenario",
                f"Design an advanced solution incorporating {', '.join(lab_concepts)}",
                f"Implement advanced features and optimizations",
                f"Handle edge cases and error conditions",
                f"Perform comprehensive testing",
                f"Compare your solution with alternative approaches",
                f"Document trade-offs and recommendations"
            ]
        else:  # expert
            title = f"{topic} - Expert Challenge (Lab {lab_num})"
            objective = f"Solve complex {topic} challenges at an expert level"
            time = 90
            instructions = [
                f"Study a real-world {topic} challenge",
                f"Research and evaluate multiple solution approaches",
                f"Design a comprehensive solution using all concepts",
                f"Implement with production-quality code",
                f"Optimize for performance, scalability, and maintainability",
                f"Create comprehensive test suite",
                f"Benchmark and profile your solution",
                f"Prepare a technical presentation of your work"
            ]
        
        lab = {
            'id': f"lab_{topic.lower().replace(' ', '_')}_{lab_num}",
            'title': title,
            'topic': topic,
            'objective': objective,
            'difficulty': difficulty,
            'estimated_time': time,
            'concepts_covered': lab_concepts,
            'prerequisites': [f"Understanding of {c}" for c in lab_concepts[:2]] if lab_concepts else [],
            'instructions': instructions,
            'expected_outcome': f"A working implementation demonstrating {topic} concepts",
            'verification_steps': [
                "Verify all requirements are met",
                "Test with provided test cases",
                "Review code quality and documentation",
                "Confirm expected behavior"
            ],
            'resources': [
                f"{topic} documentation",
                "Code examples and templates",
                "Best practices guide"
            ],
            'submission_required': True,
            'graded': False,  # Labs are practice, not graded
            'generated_at': datetime.now().isoformat()
        }
        
        return lab
    
    def _parse_llm_lab(self, content: str, topic: str, lab_num: int) -> Dict[str, Any]:
        """Parse LLM response into structured lab."""
        lab = {
            'id': f"lab_{topic.lower().replace(' ', '_')}_{lab_num}",
            'topic': topic,
            'instructions': [],
            'prerequisites': [],
            'verification_steps': [],
            'resources': [],
            'submission_required': True,
            'graded': False,
            'generated_at': datetime.now().isoformat()
        }
        
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith('TITLE:'):
                lab['title'] = line.replace('TITLE:', '').strip()
            elif line.startswith('OBJECTIVE:'):
                lab['objective'] = line.replace('OBJECTIVE:', '').strip()
            elif line.startswith('DIFFICULTY:'):
                lab['difficulty'] = line.replace('DIFFICULTY:', '').strip().lower()
            elif line.startswith('ESTIMATED_TIME:'):
                time_str = line.replace('ESTIMATED_TIME:', '').strip()
                try:
                    lab['estimated_time'] = int(''.join(filter(str.isdigit, time_str)))
                except:
                    lab['estimated_time'] = 45
            elif line.startswith('PREREQUISITES:'):
                current_section = 'prerequisites'
            elif line.startswith('INSTRUCTIONS:'):
                current_section = 'instructions'
            elif line.startswith('EXPECTED_OUTCOME:'):
                lab['expected_outcome'] = line.replace('EXPECTED_OUTCOME:', '').strip()
                current_section = None
            elif line.startswith('VERIFICATION:'):
                lab['verification_steps'].append(line.replace('VERIFICATION:', '').strip())
                current_section = None
            elif line and current_section == 'instructions':
                # Remove numbering if present
                instruction = line.lstrip('0123456789. ')
                if instruction:
                    lab['instructions'].append(instruction)
            elif line and current_section == 'prerequisites':
                lab['prerequisites'].append(line.lstrip('- '))
        
        # Set defaults if not parsed
        if 'title' not in lab:
            lab['title'] = f"{topic} - Lab {lab_num}"
        if 'objective' not in lab:
            lab['objective'] = f"Practice {topic} concepts"
        if 'difficulty' not in lab:
            lab['difficulty'] = 'intermediate'
        if 'estimated_time' not in lab:
            lab['estimated_time'] = 45
        if 'expected_outcome' not in lab:
            lab['expected_outcome'] = "Completed lab exercise"
        
        return lab


# Global instance
_lab_generator = None


def get_lab_generator(use_llm: bool = False) -> LabGenerator:
    """
    Get the global LabGenerator instance (singleton pattern).
    
    Args:
        use_llm: Whether to use actual LLM
        
    Returns:
        LabGenerator instance
    """
    global _lab_generator
    if _lab_generator is None:
        _lab_generator = LabGenerator(use_llm=use_llm)
    return _lab_generator

