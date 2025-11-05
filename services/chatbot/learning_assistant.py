# services/chatbot/learning_assistant.py

"""
Learning Assistant Chatbot

This chatbot helps students learn without providing direct answers.
It guides students through the learning process using the Socratic method.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import random


class LearningAssistant:
    """
    AI Learning Assistant that helps students without giving direct answers.
    
    Features:
    - Socratic questioning method
    - Hints and guidance instead of answers
    - Encourages critical thinking
    - Tracks conversation history
    """
    
    # Prohibited patterns - things the chatbot should NOT do
    PROHIBITED_PATTERNS = [
        "the answer is",
        "the correct answer",
        "the solution is",
        "here's the answer",
        "it's option",
        "choose option",
        "select answer"
    ]
    
    # Guidance templates
    GUIDANCE_TEMPLATES = {
        'hint': [
            "Think about {concept}. What do you already know about it?",
            "Let's break this down. What are the key components of {concept}?",
            "Consider the relationship between {concept} and {topic}. How might they connect?",
            "What have you learned about {concept} in the materials you've read?"
        ],
        'clarification': [
            "Can you explain what you understand so far about {concept}?",
            "What part of {concept} is confusing you?",
            "Let's focus on one aspect at a time. Which part would you like to explore first?",
            "What specific question do you have about {concept}?"
        ],
        'encouragement': [
            "You're on the right track! Keep thinking about it.",
            "Good question! That shows you're thinking critically.",
            "I can see you're making progress. What else can you infer?",
            "Excellent observation! How can you apply that understanding?"
        ],
        'redirect': [
            "Instead of looking for the answer, try to understand the concept first.",
            "Let's review the learning materials together. What did they say about {concept}?",
            "Think about the examples you've seen. How do they relate to this question?",
            "What would happen if you tried each option? Can you reason through them?"
        ]
    }
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the Learning Assistant.
        
        Args:
            use_llm: Whether to use actual LLM (requires OpenAI API key)
        """
        self.use_llm = use_llm
        self.llm_client = None
        self.conversation_history = {}  # student_id -> list of messages
        
        if use_llm:
            try:
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key)
                    print("✅ LLM Learning Assistant initialized")
                else:
                    print("⚠️  No OpenAI API key found, using template-based assistant")
                    self.use_llm = False
            except ImportError:
                print("⚠️  OpenAI package not installed, using template-based assistant")
                self.use_llm = False
    
    def get_response(self,
                    student_id: str,
                    message: str,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a response from the learning assistant.
        
        Args:
            student_id: Student ID
            message: Student's message
            context: Optional context (current topic, concept, etc.)
            
        Returns:
            Dictionary with response and metadata
        """
        # Initialize conversation history
        if student_id not in self.conversation_history:
            self.conversation_history[student_id] = []
        
        # Add user message to history
        self.conversation_history[student_id].append({
            'role': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        if self.use_llm and self.llm_client:
            response = self._generate_llm_response(student_id, message, context)
        else:
            response = self._generate_template_response(student_id, message, context)
        
        # Add assistant response to history
        self.conversation_history[student_id].append({
            'role': 'assistant',
            'message': response['message'],
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _generate_llm_response(self,
                              student_id: str,
                              message: str,
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using LLM with strict instructions."""
        # Create system prompt that prevents giving answers
        system_prompt = """You are a Socratic learning assistant. Your role is to help students learn through questioning and guidance, NOT by providing direct answers.

STRICT RULES:
1. NEVER provide direct answers to questions
2. NEVER tell students which option is correct
3. NEVER solve problems for students
4. ALWAYS guide students to think critically
5. ALWAYS use questions to help students discover answers themselves
6. ALWAYS encourage students to review learning materials
7. ALWAYS provide hints and guidance, not solutions

When a student asks for an answer:
- Ask them what they understand so far
- Guide them to relevant concepts
- Help them break down the problem
- Encourage them to reason through options
- Suggest reviewing specific learning materials

Be supportive, encouraging, and patient, but never give away answers."""

        # Build conversation context
        recent_history = self.conversation_history[student_id][-5:]  # Last 5 messages
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in recent_history:
            messages.append({
                "role": msg['role'],
                "content": msg['message']
            })
        
        # Add context if available
        if context:
            context_str = f"\nCurrent context: Topic: {context.get('topic', 'N/A')}, Concept: {context.get('concept', 'N/A')}"
            messages.append({
                "role": "system",
                "content": context_str
            })
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content
            
            # Double-check that response doesn't contain prohibited patterns
            if self._contains_prohibited_content(response_text):
                # Fallback to safe template response
                return self._generate_template_response(student_id, message, context)
            
            return {
                'message': response_text,
                'type': 'guidance',
                'method': 'llm'
            }
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self._generate_template_response(student_id, message, context)
    
    def _generate_template_response(self,
                                   student_id: str,
                                   message: str,
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using templates (safe fallback)."""
        message_lower = message.lower()
        
        # Extract context
        concept = context.get('concept', 'this concept') if context else 'this concept'
        topic = context.get('topic', 'this topic') if context else 'this topic'
        
        # Detect intent
        if any(word in message_lower for word in ['answer', 'solution', 'correct', 'which option']):
            # Student is asking for the answer - redirect
            response_type = 'redirect'
            templates = self.GUIDANCE_TEMPLATES['redirect']
        elif any(word in message_lower for word in ['help', 'stuck', 'confused', "don't understand"]):
            # Student needs help - provide guidance
            response_type = 'hint'
            templates = self.GUIDANCE_TEMPLATES['hint']
        elif '?' in message:
            # Student has a question - ask for clarification
            response_type = 'clarification'
            templates = self.GUIDANCE_TEMPLATES['clarification']
        else:
            # General encouragement
            response_type = 'encouragement'
            templates = self.GUIDANCE_TEMPLATES['encouragement']
        
        # Select and format template
        template = random.choice(templates)
        response_text = template.format(concept=concept, topic=topic)
        
        # Add additional guidance
        if response_type == 'redirect':
            response_text += "\n\nRemember: The goal is to understand the concept, not just find the right answer. Try reviewing the learning materials and thinking through each option carefully."
        elif response_type == 'hint':
            response_text += "\n\nTake your time to think it through. You can always review the learning materials if you need more information."
        
        return {
            'message': response_text,
            'type': response_type,
            'method': 'template'
        }
    
    def _contains_prohibited_content(self, text: str) -> bool:
        """Check if response contains prohibited patterns."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.PROHIBITED_PATTERNS)
    
    def get_conversation_history(self, student_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a student."""
        return self.conversation_history.get(student_id, [])
    
    def clear_conversation_history(self, student_id: str):
        """Clear conversation history for a student."""
        if student_id in self.conversation_history:
            self.conversation_history[student_id] = []
    
    def get_contextual_help(self,
                           concept_name: str,
                           topic_name: str,
                           help_type: str = 'general') -> str:
        """
        Get contextual help for a specific concept.
        
        Args:
            concept_name: Name of the concept
            topic_name: Name of the topic
            help_type: Type of help (general, definition, example, application)
            
        Returns:
            Help message
        """
        if help_type == 'definition':
            return f"To understand {concept_name}, start by reviewing its definition in the learning materials. What are the key characteristics that define it?"
        elif help_type == 'example':
            return f"Look at the examples provided for {concept_name}. How do they illustrate the concept? Can you think of similar examples?"
        elif help_type == 'application':
            return f"Think about when you would use {concept_name} in {topic_name}. What problems does it solve? What are its advantages?"
        else:
            return f"Let's explore {concept_name} together. What aspect of it would you like to understand better? Start by reviewing the learning materials and identifying what's unclear."


# Global instance
_learning_assistant = None


def get_learning_assistant(use_llm: bool = False) -> LearningAssistant:
    """
    Get the global LearningAssistant instance (singleton pattern).
    
    Args:
        use_llm: Whether to use actual LLM
        
    Returns:
        LearningAssistant instance
    """
    global _learning_assistant
    if _learning_assistant is None:
        _learning_assistant = LearningAssistant(use_llm=use_llm)
    return _learning_assistant

