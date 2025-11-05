# services/cognitive_diagnosis/services/assessment_engine.py

"""
Assessment Engine with LLM Question Generation and Cognitive Diagnosis

This service handles:
1. Pre-topic assessment generation
2. LLM-based question generation (3-5 questions per concept)
3. Cognitive diagnosis with guessing and slipping parameters
4. Adaptive learning path generation based on assessment results
"""

from typing import Dict, List, Any, Tuple, Optional
import random
import json
from datetime import datetime
import numpy as np


class CognitiveDiagnosisModel:
    """
    Implements cognitive diagnosis with guessing and slipping parameters.
    
    This model accounts for:
    - Guessing: Probability of answering correctly without mastery
    - Slipping: Probability of answering incorrectly despite having mastery
    """
    
    def __init__(self, guessing_prob: float = 0.25, slipping_prob: float = 0.15):
        """
        Initialize the cognitive diagnosis model.
        
        Args:
            guessing_prob: Probability of guessing correctly (default 0.25 for 4-option MCQ)
            slipping_prob: Probability of slipping (making a mistake despite knowledge)
        """
        self.guessing_prob = guessing_prob
        self.slipping_prob = slipping_prob
    
    def estimate_mastery(self, correct_answers: int, total_questions: int) -> float:
        """
        Estimate true mastery level accounting for guessing and slipping.
        
        Uses the formula:
        P(correct) = P(mastery) * (1 - slip) + (1 - P(mastery)) * guess
        
        Solving for P(mastery):
        P(mastery) = (P(correct) - guess) / (1 - slip - guess)
        
        Args:
            correct_answers: Number of correct answers
            total_questions: Total number of questions
            
        Returns:
            Estimated mastery level (0.0 to 1.0)
        """
        if total_questions == 0:
            return 0.0
        
        observed_accuracy = correct_answers / total_questions
        
        # Adjust for guessing and slipping
        numerator = observed_accuracy - self.guessing_prob
        denominator = (1 - self.slipping_prob - self.guessing_prob)
        
        if denominator <= 0:
            return observed_accuracy
        
        estimated_mastery = numerator / denominator
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, estimated_mastery))
    
    def estimate_concept_mastery(self, concept_results: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """
        Estimate mastery for multiple concepts.
        
        Args:
            concept_results: Dictionary mapping concept_name to (correct, total) tuple
            
        Returns:
            Dictionary mapping concept_name to estimated mastery level
        """
        mastery_levels = {}
        
        for concept, (correct, total) in concept_results.items():
            mastery_levels[concept] = self.estimate_mastery(correct, total)
        
        return mastery_levels


class LLMQuestionGenerator:
    """
    Generates assessment questions using LLM (simulated for now).
    
    In production, this would use OpenAI API or similar to generate questions.
    For this implementation, we'll use template-based generation with randomization.
    """
    
    def __init__(self):
        """Initialize the question generator."""
        self.question_templates = self._load_question_templates()
    
    def _load_question_templates(self) -> Dict[str, List[Dict]]:
        """Load question templates for different concept types."""
        return {
            "definition": [
                {
                    "template": "What is the definition of {concept}?",
                    "type": "multiple_choice"
                },
                {
                    "template": "Which of the following best describes {concept}?",
                    "type": "multiple_choice"
                }
            ],
            "application": [
                {
                    "template": "In which scenario would you use {concept}?",
                    "type": "multiple_choice"
                },
                {
                    "template": "How would you apply {concept} to solve {problem}?",
                    "type": "multiple_choice"
                }
            ],
            "comparison": [
                {
                    "template": "What is the main difference between {concept} and {related_concept}?",
                    "type": "multiple_choice"
                },
                {
                    "template": "When would you choose {concept} over {related_concept}?",
                    "type": "multiple_choice"
                }
            ],
            "analysis": [
                {
                    "template": "What are the advantages of using {concept}?",
                    "type": "multiple_choice"
                },
                {
                    "template": "What are the limitations of {concept}?",
                    "type": "multiple_choice"
                }
            ]
        }
    
    def generate_questions_for_concept(self, concept_name: str, concept_definition: str,
                                      keywords: List[str], num_questions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate questions for a specific concept.
        
        Args:
            concept_name: Name of the concept
            concept_definition: Definition/description of the concept
            keywords: Keywords associated with the concept
            num_questions: Number of questions to generate (default 3)
            
        Returns:
            List of question dictionaries
        """
        questions = []
        question_types = list(self.question_templates.keys())
        
        for i in range(num_questions):
            # Select a random question type
            q_type = random.choice(question_types)
            template = random.choice(self.question_templates[q_type])
            
            # Generate question text
            question_text = template["template"].format(
                concept=concept_name,
                related_concept=random.choice(keywords) if keywords else "related concept",
                problem="a real-world problem"
            )
            
            # Generate plausible options
            options = self._generate_options(concept_name, concept_definition, keywords)
            correct_answer = 0  # First option is correct
            
            # Shuffle options
            shuffled_options, new_correct = self._shuffle_options(options, correct_answer)
            
            question = {
                "question_id": f"q_{concept_name}_{i}_{random.randint(1000, 9999)}",
                "concept": concept_name,
                "text": question_text,
                "options": shuffled_options,
                "correct_answer": new_correct,
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "question_type": q_type,
                "generated_by_llm": True,
                "created_at": datetime.now().isoformat()
            }
            
            questions.append(question)
        
        return questions
    
    def _generate_options(self, concept_name: str, definition: str, keywords: List[str]) -> List[str]:
        """
        Generate plausible answer options.
        
        Args:
            concept_name: Name of the concept
            definition: Definition of the concept
            keywords: Keywords associated with the concept
            
        Returns:
            List of 4 options (first one is correct)
        """
        # Correct answer (simplified from definition)
        correct = definition[:100] + "..." if len(definition) > 100 else definition
        
        # Generate distractors
        distractors = [
            f"A method that is unrelated to {concept_name}",
            f"An approach that contradicts the principles of {concept_name}",
            f"A technique that is similar but not the same as {concept_name}"
        ]
        
        return [correct] + distractors
    
    def _shuffle_options(self, options: List[str], correct_index: int) -> Tuple[List[str], int]:
        """
        Shuffle options and return new correct answer index.
        
        Args:
            options: List of options
            correct_index: Index of correct answer before shuffling
            
        Returns:
            Tuple of (shuffled_options, new_correct_index)
        """
        correct_option = options[correct_index]
        shuffled = options.copy()
        random.shuffle(shuffled)
        new_correct_index = shuffled.index(correct_option)
        
        return shuffled, new_correct_index


class AssessmentEngine:
    """
    Main assessment engine that orchestrates question generation and cognitive diagnosis.
    """
    
    def __init__(self, dynamic_graph_manager=None):
        """
        Initialize the assessment engine.
        
        Args:
            dynamic_graph_manager: Instance of DynamicGraphManager for data persistence
        """
        self.graph_manager = dynamic_graph_manager
        self.question_generator = LLMQuestionGenerator()
        self.cognitive_model = CognitiveDiagnosisModel()
        self.question_bank = {}  # Cache for generated questions
    
    def generate_pre_topic_assessment(self, topic_name: str, concepts: List[Dict[str, Any]],
                                     questions_per_concept: int = 3) -> Dict[str, Any]:
        """
        Generate a pre-topic assessment with questions for each concept.
        
        Args:
            topic_name: Name of the topic
            concepts: List of concept dictionaries with name, definition, keywords
            questions_per_concept: Number of questions to generate per concept (default 3)
            
        Returns:
            Dictionary containing assessment information and questions
        """
        all_questions = []
        concept_question_map = {}
        
        # Generate questions for each concept
        for concept in concepts:
            concept_name = concept.get('name', 'Unknown')
            concept_def = concept.get('definition', '')
            keywords = concept.get('keywords', [])
            
            # Generate questions
            questions = self.question_generator.generate_questions_for_concept(
                concept_name, concept_def, keywords, questions_per_concept
            )
            
            # Store in question bank
            for q in questions:
                self.question_bank[q['question_id']] = q
                
                # Store in graph if available
                if self.graph_manager:
                    self.graph_manager.create_question(
                        q['question_id'], concept_name, q['text'],
                        q['options'], q['correct_answer'], q['difficulty']
                    )
            
            concept_question_map[concept_name] = [q['question_id'] for q in questions]
            all_questions.extend(questions)
        
        # Randomly select 10 questions for the assessment
        selected_questions = random.sample(all_questions, min(10, len(all_questions)))
        
        assessment_id = f"pre_topic_{topic_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create assessment in graph
        if self.graph_manager:
            self.graph_manager.create_assessment(
                assessment_id, topic_name, "pre_topic",
                [q['question_id'] for q in selected_questions]
            )
        
        return {
            "assessment_id": assessment_id,
            "topic_name": topic_name,
            "type": "pre_topic",
            "questions": selected_questions,
            "total_questions": len(selected_questions),
            "concept_question_map": concept_question_map,
            "created_at": datetime.now().isoformat()
        }

    def evaluate_assessment(self, assessment_id: str, student_id: str,
                           student_answers: Dict[str, int]) -> Dict[str, Any]:
        """
        Evaluate a student's assessment and estimate mastery levels.

        Args:
            assessment_id: ID of the assessment
            student_id: ID of the student
            student_answers: Dictionary mapping question_id to selected answer index

        Returns:
            Dictionary containing evaluation results and mastery estimates
        """
        # Get assessment questions
        assessment_questions = []
        for q_id in student_answers.keys():
            if q_id in self.question_bank:
                assessment_questions.append(self.question_bank[q_id])

        # Calculate results per concept
        concept_results = {}
        total_correct = 0
        total_questions = len(assessment_questions)

        for question in assessment_questions:
            q_id = question['question_id']
            concept = question['concept']
            correct_answer = question['correct_answer']
            student_answer = student_answers.get(q_id, -1)

            is_correct = (student_answer == correct_answer)

            if concept not in concept_results:
                concept_results[concept] = {'correct': 0, 'total': 0}

            concept_results[concept]['total'] += 1
            if is_correct:
                concept_results[concept]['correct'] += 1
                total_correct += 1

        # Estimate mastery levels using cognitive diagnosis
        concept_mastery = {}
        for concept, results in concept_results.items():
            mastery = self.cognitive_model.estimate_mastery(
                results['correct'], results['total']
            )
            concept_mastery[concept] = mastery

            # Update in graph
            if self.graph_manager:
                self.graph_manager.update_student_mastery(student_id, concept, mastery)

        # Calculate overall score
        overall_score = total_correct / max(total_questions, 1)

        # Record assessment attempt
        if self.graph_manager:
            self.graph_manager.record_assessment_attempt(
                student_id, assessment_id, overall_score, student_answers
            )

        return {
            "assessment_id": assessment_id,
            "student_id": student_id,
            "overall_score": round(overall_score, 2),
            "total_correct": total_correct,
            "total_questions": total_questions,
            "concept_mastery": concept_mastery,
            "concept_results": concept_results,
            "evaluated_at": datetime.now().isoformat()
        }

    def generate_adaptive_learning_path(self, student_id: str, topic_name: str,
                                       assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an adaptive learning path based on assessment results.

        High score (>80%): Generate advanced/challenging content
        Low score (<50%): Recommend supplementary resources

        Args:
            student_id: ID of the student
            topic_name: Name of the topic
            assessment_results: Results from evaluate_assessment

        Returns:
            Dictionary containing adaptive learning path
        """
        overall_score = assessment_results.get('overall_score', 0)
        concept_mastery = assessment_results.get('concept_mastery', {})

        learning_path = {
            "student_id": student_id,
            "topic_name": topic_name,
            "overall_score": overall_score,
            "path_type": "",
            "recommendations": [],
            "next_steps": [],
            "created_at": datetime.now().isoformat()
        }

        if overall_score >= 0.8:
            # High score: Generate advanced content
            learning_path["path_type"] = "advanced"
            learning_path["recommendations"] = self._generate_advanced_content(topic_name, concept_mastery)
            learning_path["next_steps"] = [
                "Explore advanced topics and applications",
                "Work on challenging projects",
                "Consider peer teaching or mentoring"
            ]
        elif overall_score < 0.5:
            # Low score: Recommend supplementary resources
            learning_path["path_type"] = "foundational"
            learning_path["recommendations"] = self._generate_supplementary_resources(topic_name, concept_mastery)
            learning_path["next_steps"] = [
                "Review foundational concepts",
                "Watch tutorial videos",
                "Practice with guided exercises",
                "Seek help from instructor or peers"
            ]
        else:
            # Medium score: Standard progression
            learning_path["path_type"] = "standard"
            learning_path["recommendations"] = self._generate_standard_content(topic_name, concept_mastery)
            learning_path["next_steps"] = [
                "Continue with regular coursework",
                "Practice weak areas",
                "Attempt practice quizzes"
            ]

        return learning_path

    def _generate_advanced_content(self, topic_name: str, concept_mastery: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate advanced content for high-performing students."""
        advanced_content = [
            {
                "type": "advanced_topic",
                "title": f"Advanced Applications of {topic_name}",
                "description": "Explore cutting-edge applications and research in this area",
                "difficulty": "hard",
                "estimated_time": "2-3 hours"
            },
            {
                "type": "project",
                "title": f"Capstone Project: {topic_name}",
                "description": "Apply your knowledge to a real-world project",
                "difficulty": "hard",
                "estimated_time": "5-10 hours"
            },
            {
                "type": "research_paper",
                "title": f"Recent Research in {topic_name}",
                "description": "Read and analyze recent academic papers",
                "difficulty": "hard",
                "estimated_time": "3-4 hours"
            }
        ]

        return advanced_content

    def _generate_supplementary_resources(self, topic_name: str, concept_mastery: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate supplementary resources for struggling students."""
        # Identify weak concepts (mastery < 0.5)
        weak_concepts = [concept for concept, mastery in concept_mastery.items() if mastery < 0.5]

        resources = []

        for concept in weak_concepts:
            resources.extend([
                {
                    "type": "video",
                    "title": f"Introduction to {concept}",
                    "description": f"Video tutorial explaining {concept} fundamentals",
                    "url": f"https://youtube.com/search?q={concept.replace(' ', '+')}+tutorial",
                    "estimated_time": "15-20 minutes",
                    "concept": concept
                },
                {
                    "type": "blog",
                    "title": f"Understanding {concept}",
                    "description": f"Comprehensive blog post about {concept}",
                    "url": f"https://medium.com/search?q={concept.replace(' ', '+')}",
                    "estimated_time": "10-15 minutes",
                    "concept": concept
                },
                {
                    "type": "practice",
                    "title": f"Practice Exercises: {concept}",
                    "description": f"Guided practice problems for {concept}",
                    "difficulty": "easy",
                    "estimated_time": "30 minutes",
                    "concept": concept
                }
            ])

        return resources[:10]  # Limit to 10 resources

    def _generate_standard_content(self, topic_name: str, concept_mastery: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate standard content for average-performing students."""
        standard_content = [
            {
                "type": "lab",
                "title": f"{topic_name} Hands-on Lab",
                "description": "Practice your skills with guided exercises",
                "difficulty": "medium",
                "estimated_time": "1-2 hours"
            },
            {
                "type": "quiz",
                "title": f"{topic_name} Practice Quiz",
                "description": "Test your understanding with practice questions",
                "difficulty": "medium",
                "estimated_time": "30 minutes"
            },
            {
                "type": "reading",
                "title": f"{topic_name} Deep Dive",
                "description": "Comprehensive reading material",
                "difficulty": "medium",
                "estimated_time": "45 minutes"
            }
        ]

        return standard_content

