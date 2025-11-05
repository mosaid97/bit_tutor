# services/content_generation/services/question_generator.py

"""
LLM Question Generator Service

This service generates assessment questions for concepts using LLM.
Questions are stored in the knowledge graph and connected to concepts.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import random


class QuestionGenerator:
    """
    Generates assessment questions for concepts using LLM or templates.
    
    Features:
    - Generates 3-5 questions per concept
    - Multiple choice format
    - Stores questions in knowledge graph
    - Uses LLM or template-based generation
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the Question Generator.
        
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
                    print("✅ LLM Question Generator initialized")
                else:
                    print("⚠️  No OpenAI API key found, using template-based question generation")
                    self.use_llm = False
            except ImportError:
                print("⚠️  OpenAI package not installed, using template-based question generation")
                self.use_llm = False
    
    def generate_questions_for_concept(self,
                                      concept_name: str,
                                      topic_name: str,
                                      theory_data: Optional[Dict[str, Any]] = None,
                                      num_questions: int = 4) -> List[Dict[str, Any]]:
        """
        Generate assessment questions for a concept.
        
        Args:
            concept_name: Name of the concept
            topic_name: Parent topic name
            theory_data: Optional theory data from lab_tutor
            num_questions: Number of questions to generate (3-5)
            
        Returns:
            List of question dictionaries
        """
        num_questions = max(3, min(5, num_questions))  # Ensure 3-5 questions
        
        if self.use_llm and self.llm_client:
            return self._generate_with_llm(concept_name, topic_name, theory_data, num_questions)
        else:
            return self._generate_with_template(concept_name, topic_name, theory_data, num_questions)
    
    def _generate_with_llm(self,
                          concept_name: str,
                          topic_name: str,
                          theory_data: Optional[Dict[str, Any]],
                          num_questions: int) -> List[Dict[str, Any]]:
        """Generate questions using actual LLM."""
        # Prepare context from theory data
        context = ""
        if theory_data:
            context = theory_data.get('compressed_text', theory_data.get('original_text', ''))
        
        # Create prompt
        prompt = self._create_llm_prompt(concept_name, topic_name, context, num_questions)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educational assessment creator. Generate challenging but fair multiple-choice questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse the response into structured questions
            return self._parse_llm_questions(content, concept_name, topic_name)
            
        except Exception as e:
            print(f"Error generating questions with LLM: {e}")
            # Fallback to template
            return self._generate_with_template(concept_name, topic_name, theory_data, num_questions)
    
    def _create_llm_prompt(self,
                          concept_name: str,
                          topic_name: str,
                          context: str,
                          num_questions: int) -> str:
        """Create prompt for LLM question generation."""
        prompt = f"""Generate {num_questions} multiple-choice assessment questions about "{concept_name}" in the context of "{topic_name}".

Context Information:
{context if context else "No additional context provided"}

Requirements:
1. Each question should test understanding, not just memorization
2. Include 4 answer options (A, B, C, D)
3. Mark the correct answer
4. Questions should be challenging but fair
5. Cover different aspects of the concept
6. Avoid questions that are too easy or too hard

Format each question as:
QUESTION: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
CORRECT: [A/B/C/D]
EXPLANATION: [brief explanation of why this is correct]

Generate {num_questions} questions now:
"""
        return prompt
    
    def _parse_llm_questions(self,
                            content: str,
                            concept_name: str,
                            topic_name: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured questions."""
        questions = []
        current_question = {}
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('QUESTION:'):
                if current_question:
                    questions.append(current_question)
                current_question = {
                    'question_text': line.replace('QUESTION:', '').strip(),
                    'options': {},
                    'concept_name': concept_name,
                    'topic_name': topic_name,
                    'difficulty': 'medium',
                    'generated_at': datetime.now().isoformat(),
                    'generation_method': 'llm'
                }
            elif line.startswith('A)'):
                current_question['options']['A'] = line[2:].strip()
            elif line.startswith('B)'):
                current_question['options']['B'] = line[2:].strip()
            elif line.startswith('C)'):
                current_question['options']['C'] = line[2:].strip()
            elif line.startswith('D)'):
                current_question['options']['D'] = line[2:].strip()
            elif line.startswith('CORRECT:'):
                current_question['correct_answer'] = line.replace('CORRECT:', '').strip()
            elif line.startswith('EXPLANATION:'):
                current_question['explanation'] = line.replace('EXPLANATION:', '').strip()
        
        if current_question and 'question_text' in current_question:
            questions.append(current_question)
        
        return questions
    
    def _generate_with_template(self,
                                concept_name: str,
                                topic_name: str,
                                theory_data: Optional[Dict[str, Any]],
                                num_questions: int) -> List[Dict[str, Any]]:
        """Generate questions using templates (fallback)."""
        questions = []
        
        # Extract keywords from theory data
        keywords = []
        if theory_data:
            keywords = theory_data.get('keywords', [])
        
        # Question templates
        templates = [
            {
                'question': f"What is the primary purpose of {concept_name} in {topic_name}?",
                'type': 'definition'
            },
            {
                'question': f"Which of the following best describes {concept_name}?",
                'type': 'description'
            },
            {
                'question': f"In the context of {topic_name}, when would you use {concept_name}?",
                'type': 'application'
            },
            {
                'question': f"What is a key characteristic of {concept_name}?",
                'type': 'characteristic'
            },
            {
                'question': f"How does {concept_name} relate to {topic_name}?",
                'type': 'relationship'
            }
        ]
        
        # Generate questions
        selected_templates = random.sample(templates, min(num_questions, len(templates)))
        
        for i, template in enumerate(selected_templates):
            question_id = f"q_{concept_name.lower().replace(' ', '_')}_{i+1}_{int(datetime.now().timestamp())}"
            question = {
                'question_id': question_id,
                'id': question_id,  # Backward compatibility
                'question': template['question'],  # Use 'question' not 'question_text'
                'question_text': template['question'],  # Keep for backward compatibility
                'options': self._generate_options(concept_name, topic_name, keywords, template['type']),
                'correct_answer': 'A',  # First option is correct by default
                'concept_name': concept_name,
                'topic_name': topic_name,
                'difficulty': random.choice(['easy', 'medium', 'hard']),
                'question_type': template['type'],
                'generated_at': datetime.now().isoformat(),
                'generation_method': 'template'
            }
            questions.append(question)
        
        return questions
    
    def _generate_options(self,
                         concept_name: str,
                         topic_name: str,
                         keywords: List[str],
                         question_type: str) -> Dict[str, str]:
        """Generate answer options for a question."""
        if question_type == 'definition':
            return {
                'A': f"A fundamental concept in {topic_name} that enables efficient data management",
                'B': f"A deprecated approach that is no longer used in modern systems",
                'C': f"A programming language feature unrelated to {topic_name}",
                'D': f"A hardware component used in database servers"
            }
        elif question_type == 'description':
            return {
                'A': f"A key component of {topic_name} that provides scalability and performance",
                'B': f"An outdated technology replaced by newer alternatives",
                'C': f"A user interface design pattern",
                'D': f"A network protocol for data transmission"
            }
        elif question_type == 'application':
            return {
                'A': f"When you need to handle large-scale data with specific requirements",
                'B': f"Only for small-scale applications with limited data",
                'C': f"Never, as it's been deprecated",
                'D': f"Only in legacy systems that cannot be updated"
            }
        elif question_type == 'characteristic':
            return {
                'A': f"High performance and scalability for specific use cases",
                'B': f"Limited functionality and poor performance",
                'C': f"Only works with specific programming languages",
                'D': f"Requires expensive proprietary hardware"
            }
        else:  # relationship
            return {
                'A': f"It is a core component that enables {topic_name} functionality",
                'B': f"It is completely unrelated to {topic_name}",
                'C': f"It conflicts with {topic_name} principles",
                'D': f"It is only used for testing purposes"
            }
    
    def store_questions_in_graph(self,
                                questions: List[Dict[str, Any]],
                                graph_manager) -> bool:
        """
        Store generated questions in the knowledge graph.

        Args:
            questions: List of question dictionaries
            graph_manager: DynamicGraphManager instance

        Returns:
            True if successful, False otherwise
        """
        if not graph_manager or not graph_manager.neo4j:
            print("⚠️  Neo4j not available, questions not stored in graph")
            return False

        try:
            for question in questions:
                query = """
                MERGE (c:Concept {name: $concept_name})
                CREATE (q:Question {
                    question_id: $question_id,
                    question: $question_text,
                    correct_answer: $correct_answer,
                    difficulty: $difficulty,
                    question_type: $question_type,
                    generated_at: $generated_at,
                    generation_method: $generation_method,
                    options: $options
                })
                CREATE (q)-[:TESTS]->(c)
                RETURN q
                """

                # Use 'question' field if available, fallback to 'question_text'
                question_text = question.get('question', question.get('question_text', ''))
                question_id = question.get('question_id', question.get('id', f"q_{question['concept_name']}_{datetime.now().timestamp()}"))

                graph_manager.neo4j.graph.query(query, {
                    'concept_name': question['concept_name'],
                    'question_id': question_id,
                    'question_text': question_text,
                    'correct_answer': question['correct_answer'],
                    'difficulty': question['difficulty'],
                    'question_type': question.get('question_type', 'general'),
                    'generated_at': question['generated_at'],
                    'generation_method': question['generation_method'],
                    'options': json.dumps(question['options'])
                })

            print(f"✅ Stored {len(questions)} questions in knowledge graph")
            return True

        except Exception as e:
            print(f"Error storing questions in graph: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global instance
_question_generator = None


def get_question_generator(use_llm: bool = False) -> QuestionGenerator:
    """
    Get the global QuestionGenerator instance (singleton pattern).
    
    Args:
        use_llm: Whether to use actual LLM
        
    Returns:
        QuestionGenerator instance
    """
    global _question_generator
    if _question_generator is None:
        _question_generator = QuestionGenerator(use_llm=use_llm)
    return _question_generator

