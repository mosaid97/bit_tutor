# services/content_generation/services/quiz_generator.py

"""
Quiz Generator Service

Generates graded quizzes for topics using LLM or templates.
Quizzes are placed at the end of each topic/chapter.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import random


class QuizGenerator:
    """
    Generates graded quiz questions for topics.
    
    Features:
    - Generates 10-15 questions per topic
    - Multiple choice format
    - Covers all concepts in the topic
    - Graded (counts toward student grade)
    - Uses LLM or template-based generation
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the Quiz Generator.
        
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
                    print("✅ LLM Quiz Generator initialized")
                else:
                    print("⚠️  No OpenAI API key found, using template-based quiz generation")
                    self.use_llm = False
            except ImportError:
                print("⚠️  OpenAI package not installed, using template-based quiz generation")
                self.use_llm = False
    
    def generate_quiz_for_topic(self,
                                topic_name: str,
                                concepts: List[str],
                                theory_data: Optional[List[Dict[str, Any]]] = None,
                                num_questions: int = 15) -> Dict[str, Any]:
        """
        Generate a graded quiz for a topic.
        
        Args:
            topic_name: Name of the topic
            concepts: List of concepts in the topic
            theory_data: Optional theory data from lab_tutor
            num_questions: Number of questions (default 15)
            
        Returns:
            Quiz dictionary with questions and metadata
        """
        num_questions = max(10, min(20, num_questions))  # 10-20 questions
        
        if self.use_llm and self.llm_client:
            questions = self._generate_with_llm(topic_name, concepts, theory_data, num_questions)
        else:
            questions = self._generate_with_template(topic_name, concepts, theory_data, num_questions)
        
        # Create quiz structure
        quiz = {
            'quiz_id': f"quiz_{topic_name.lower().replace(' ', '_')}_{datetime.now().timestamp()}",
            'topic_name': topic_name,
            'title': f"{topic_name} - Graded Quiz",
            'description': f"Test your understanding of {topic_name}. This quiz counts toward your grade.",
            'questions': questions,
            'total_questions': len(questions),
            'passing_score': 0.7,  # 70% to pass
            'time_limit': len(questions) * 90,  # 90 seconds per question
            'is_graded': True,
            'generated_at': datetime.now().isoformat()
        }
        
        return quiz
    
    def _generate_with_llm(self,
                          topic_name: str,
                          concepts: List[str],
                          theory_data: Optional[List[Dict[str, Any]]],
                          num_questions: int) -> List[Dict[str, Any]]:
        """Generate quiz questions using LLM."""
        # Similar to question_generator but for comprehensive quiz
        questions = []
        
        # Distribute questions across concepts
        questions_per_concept = max(1, num_questions // len(concepts))
        
        for concept in concepts:
            # Find theory for this concept
            theory_text = ""
            if theory_data:
                for theory in theory_data:
                    if concept in theory.get('keywords', []):
                        theory_text = theory.get('compressed_text', '')
                        break
            
            prompt = f"""Generate {questions_per_concept} challenging quiz questions about "{concept}" in the context of "{topic_name}".

Context: {theory_text if theory_text else 'No additional context'}

Requirements:
1. Questions should test deep understanding, not just memorization
2. Include 4 answer options (A, B, C, D)
3. Mark the correct answer
4. Questions should be at intermediate to advanced level
5. Cover different aspects: definition, application, comparison, problem-solving

Format each question as:
QUESTION: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
CORRECT: [A/B/C/D]
EXPLANATION: [why this is correct]
"""
            
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert educational quiz creator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Parse response
                parsed = self._parse_llm_response(response.choices[0].message.content, concept, topic_name)
                questions.extend(parsed)
                
            except Exception as e:
                print(f"Error generating LLM quiz questions: {e}")
                # Fallback to template
                questions.extend(self._generate_template_questions(concept, topic_name, questions_per_concept))
        
        return questions[:num_questions]  # Ensure we don't exceed limit
    
    def _generate_with_template(self,
                                topic_name: str,
                                concepts: List[str],
                                theory_data: Optional[List[Dict[str, Any]]],
                                num_questions: int) -> List[Dict[str, Any]]:
        """Generate quiz questions using templates."""
        questions = []

        # Handle empty concepts list
        if not concepts or len(concepts) == 0:
            # Use topic name as a fallback concept
            concepts = [topic_name]

        # Distribute questions across concepts
        questions_per_concept = max(1, num_questions // len(concepts))
        
        for concept in concepts:
            concept_questions = self._generate_template_questions(concept, topic_name, questions_per_concept)
            questions.extend(concept_questions)
        
        return questions[:num_questions]
    
    def _generate_template_questions(self, concept: str, topic: str, count: int) -> List[Dict[str, Any]]:
        """Generate template-based questions for a concept."""
        templates = [
            {
                'question': f"Which of the following best describes {concept} in {topic}?",
                'type': 'definition',
                'difficulty': 'medium'
            },
            {
                'question': f"What is the primary advantage of using {concept} in {topic}?",
                'type': 'advantage',
                'difficulty': 'medium'
            },
            {
                'question': f"In which scenario would you use {concept}?",
                'type': 'application',
                'difficulty': 'hard'
            },
            {
                'question': f"How does {concept} differ from other approaches in {topic}?",
                'type': 'comparison',
                'difficulty': 'hard'
            },
            {
                'question': f"What is a key characteristic of {concept}?",
                'type': 'characteristic',
                'difficulty': 'easy'
            }
        ]
        
        selected = random.sample(templates, min(count, len(templates)))
        questions = []
        
        for i, template in enumerate(selected):
            question = {
                'id': f"q_{concept.lower().replace(' ', '_')}_{i+1}",
                'question_text': template['question'],
                'options': self._generate_options(concept, topic, template['type']),
                'correct_answer': 'A',
                'concept': concept,
                'topic': topic,
                'difficulty': template['difficulty'],
                'question_type': template['type'],
                'points': 1
            }
            questions.append(question)
        
        return questions
    
    def _generate_options(self, concept: str, topic: str, q_type: str) -> Dict[str, str]:
        """Generate answer options based on question type."""
        if q_type == 'definition':
            return {
                'A': f"A scalable and efficient approach for handling {topic} requirements",
                'B': f"An outdated method that is no longer recommended",
                'C': f"A programming language feature unrelated to {topic}",
                'D': f"A hardware-specific implementation"
            }
        elif q_type == 'advantage':
            return {
                'A': f"High performance and flexibility for specific use cases",
                'B': f"Lower cost but reduced functionality",
                'C': f"Easier to learn but limited scalability",
                'D': f"Better for small projects only"
            }
        elif q_type == 'application':
            return {
                'A': f"When you need to handle large-scale data with specific requirements",
                'B': f"Only for legacy systems that cannot be updated",
                'C': f"Never, as better alternatives exist",
                'D': f"Only in academic research, not production"
            }
        elif q_type == 'comparison':
            return {
                'A': f"It provides better scalability and performance for certain workloads",
                'B': f"It is exactly the same as other approaches",
                'C': f"It is always inferior to alternatives",
                'D': f"It only works with specific programming languages"
            }
        else:  # characteristic
            return {
                'A': f"Optimized for specific data access patterns and scalability",
                'B': f"Requires expensive proprietary hardware",
                'C': f"Only supports a single programming language",
                'D': f"Cannot handle concurrent users"
            }
    
    def _parse_llm_response(self, content: str, concept: str, topic: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured questions."""
        questions = []
        current_q = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith('QUESTION:'):
                if current_q:
                    questions.append(current_q)
                current_q = {
                    'question_text': line.replace('QUESTION:', '').strip(),
                    'options': {},
                    'concept': concept,
                    'topic': topic,
                    'points': 1
                }
            elif line.startswith('A)'):
                current_q['options']['A'] = line[2:].strip()
            elif line.startswith('B)'):
                current_q['options']['B'] = line[2:].strip()
            elif line.startswith('C)'):
                current_q['options']['C'] = line[2:].strip()
            elif line.startswith('D)'):
                current_q['options']['D'] = line[2:].strip()
            elif line.startswith('CORRECT:'):
                current_q['correct_answer'] = line.replace('CORRECT:', '').strip()
            elif line.startswith('EXPLANATION:'):
                current_q['explanation'] = line.replace('EXPLANATION:', '').strip()
        
        if current_q and 'question_text' in current_q:
            questions.append(current_q)
        
        return questions


# Global instance
_quiz_generator = None


def get_quiz_generator(use_llm: bool = False) -> QuizGenerator:
    """
    Get the global QuizGenerator instance (singleton pattern).
    
    Args:
        use_llm: Whether to use actual LLM
        
    Returns:
        QuizGenerator instance
    """
    global _quiz_generator
    if _quiz_generator is None:
        _quiz_generator = QuizGenerator(use_llm=use_llm)
    return _quiz_generator

