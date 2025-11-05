"""
Lab Agent Service

Provides AI-powered hints and guidance for lab exercises.
The agent helps students without giving direct answers.
"""

import os
from openai import OpenAI

class LabAgentService:
    """AI agent that provides hints for lab exercises"""
    
    def __init__(self):
        """Initialize the lab agent with OpenAI API"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"
    
    def get_hint(self, concept_name, exercise_description, student_code, hint_level=1):
        """
        Get a hint for a lab exercise without giving the direct answer.
        
        Args:
            concept_name: The concept being practiced
            exercise_description: Description of the exercise
            student_code: Current code written by student
            hint_level: 1 (gentle), 2 (moderate), 3 (detailed)
        
        Returns:
            dict: Hint response with guidance
        """
        
        # Build prompt based on hint level
        if hint_level == 1:
            hint_type = "a gentle nudge in the right direction"
        elif hint_level == 2:
            hint_type = "a moderate hint with some specific guidance"
        else:
            hint_type = "a detailed hint with step-by-step guidance (but not the complete solution)"
        
        prompt = f"""You are a helpful lab assistant for a Big Data and NoSQL course. 
A student is working on an exercise about: {concept_name}

Exercise Description:
{exercise_description}

Student's Current Code:
```python
{student_code if student_code else "# No code written yet"}
```

Provide {hint_type} to help the student solve this exercise.

IMPORTANT RULES:
1. DO NOT provide the complete solution or direct answer
2. Guide the student's thinking process
3. Ask leading questions when appropriate
4. Point out what they're doing right
5. Suggest what to think about next
6. Reference the concept they're learning
7. Keep hints concise and encouraging

Respond in a friendly, encouraging tone."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful lab teaching assistant who guides students without giving direct answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            hint_text = response.choices[0].message.content
            
            return {
                'success': True,
                'hint': hint_text,
                'hint_level': hint_level
            }
            
        except Exception as e:
            print(f"Error getting hint: {e}")
            return {
                'success': False,
                'error': str(e),
                'hint': "I'm having trouble generating a hint right now. Try reviewing the concept description above."
            }
    
    def check_code(self, concept_name, exercise_description, student_code, expected_output=None):
        """
        Check student's code and provide feedback without revealing the answer.
        
        Args:
            concept_name: The concept being practiced
            exercise_description: Description of the exercise
            student_code: Code written by student
            expected_output: Optional expected output description
        
        Returns:
            dict: Feedback on the code
        """
        
        prompt = f"""You are a code reviewer for a Big Data and NoSQL lab exercise.

Concept: {concept_name}

Exercise:
{exercise_description}

Student's Code:
```python
{student_code}
```

{f"Expected Output: {expected_output}" if expected_output else ""}

Provide constructive feedback on the student's code:
1. What they did well
2. Any errors or issues (without fixing them directly)
3. Suggestions for improvement
4. Whether they're on the right track

DO NOT provide the corrected code. Guide them to find and fix issues themselves.

Keep feedback encouraging and educational."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful code reviewer who provides guidance without giving direct solutions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            feedback = response.choices[0].message.content
            
            return {
                'success': True,
                'feedback': feedback
            }
            
        except Exception as e:
            print(f"Error checking code: {e}")
            return {
                'success': False,
                'error': str(e),
                'feedback': "Unable to check code at this time."
            }
    
    def explain_concept(self, concept_name, concept_description, context="lab exercise"):
        """
        Explain a concept in the context of the lab.
        
        Args:
            concept_name: Name of the concept
            concept_description: Description of the concept
            context: Context for the explanation
        
        Returns:
            dict: Explanation of the concept
        """
        
        prompt = f"""Explain the following concept in the context of a {context}:

Concept: {concept_name}
Description: {concept_description}

Provide:
1. A clear, concise explanation
2. Why it's important in Big Data/NoSQL
3. A simple example or analogy
4. How it applies to practical scenarios

Keep it brief (3-4 paragraphs) and student-friendly."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable instructor explaining Big Data and NoSQL concepts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            explanation = response.choices[0].message.content
            
            return {
                'success': True,
                'explanation': explanation
            }
            
        except Exception as e:
            print(f"Error explaining concept: {e}")
            return {
                'success': False,
                'error': str(e),
                'explanation': concept_description
            }
    
    def generate_exercise_hint(self, concept_name, difficulty="medium"):
        """
        Generate a practice exercise hint for a concept.
        
        Args:
            concept_name: The concept to practice
            difficulty: easy, medium, or hard
        
        Returns:
            dict: Exercise suggestion
        """
        
        prompt = f"""Suggest a {difficulty} practice exercise for the concept: {concept_name}

The exercise should:
1. Be practical and hands-on
2. Reinforce understanding of {concept_name}
3. Be appropriate for a Big Data/NoSQL course
4. Include a brief description and what the student should accomplish

Format:
- Title: [Exercise title]
- Description: [What to do]
- Goal: [What they'll learn]
- Starter hint: [Where to begin]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative instructor designing lab exercises."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            
            exercise = response.choices[0].message.content
            
            return {
                'success': True,
                'exercise': exercise
            }
            
        except Exception as e:
            print(f"Error generating exercise: {e}")
            return {
                'success': False,
                'error': str(e),
                'exercise': f"Practice implementing {concept_name} in a real-world scenario."
            }

