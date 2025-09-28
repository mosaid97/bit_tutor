# services/cognitive_diagnosis/models/cold_start_assessor.py

import random

class LLM_Cold_Start_Assessor:
    """Simulates an LLM providing a zero-shot diagnosis for a new student."""
    def __init__(self, all_kcs):
        self.all_kcs = all_kcs
        self.possible_hobbies = [
            "sports", "music", "gaming", "cooking", "art", 
            "science", "space", "history", "travel", "nature",
            "technology", "movies", "animals", "books", "fashion"
        ]
        print("\n--- Building Cognitive Diagnosis & XAI Components ---")
        print("Initialized LLM Cold-Start Assessor.")

    def get_initial_diagnosis(self, student_id):
        """
        Creates a personalized initial diagnosis for each student.
        Each student gets a unique profile with different strengths and weaknesses.
        Also assigns hobbies to personalize learning content.
        """
        print(f"LLM generating initial diagnosis for new student: {student_id}...")
        
        # Set random seed based on student_id for reproducibility but unique profiles
        random.seed(hash(student_id) % 10000)
        
        # Create a base profile with low mastery levels
        profile = {kc: round(random.uniform(0.05, 0.25), 2) for kc in self.all_kcs}
        
        # Customize based on student ID to create distinct profiles
        if 'alice' in student_id.lower():
            # Alice has strengths in variables and printing, weakness in loops
            for kc in self.all_kcs:
                if 'KC1' in kc or 'KC4' in kc:  # Variables or Printing
                    profile[kc] = round(random.uniform(0.30, 0.45), 2)
                elif 'KC6' in kc:  # For Loops
                    profile[kc] = round(random.uniform(0.05, 0.15), 2)
        
        elif 'bob' in student_id.lower():
            # Bob has strengths in loops and functions, weakness in conditionals
            for kc in self.all_kcs:
                if 'KC6' in kc or 'KC8' in kc:  # For Loops or Functions
                    profile[kc] = round(random.uniform(0.35, 0.50), 2)
                elif 'KC5' in kc:  # Conditionals
                    profile[kc] = round(random.uniform(0.05, 0.15), 2)
        
        elif 'charlie' in student_id.lower():
            # Charlie has strengths in data types and conditionals, weakness in lists and functions
            for kc in self.all_kcs:
                if 'KC2' in kc or 'KC5' in kc:  # Data Types or Conditionals
                    profile[kc] = round(random.uniform(0.30, 0.45), 2)
                elif 'KC7' in kc or 'KC8' in kc:  # Lists or Functions
                    profile[kc] = round(random.uniform(0.05, 0.15), 2)
        
        else:
            # For any other student, create a random profile with 2 strengths and 2 weaknesses
            strengths = random.sample(list(self.all_kcs), 2)
            remaining = [kc for kc in self.all_kcs if kc not in strengths]
            weaknesses = random.sample(remaining, 2)
            
            for kc in strengths:
                profile[kc] = round(random.uniform(0.30, 0.50), 2)
            
            for kc in weaknesses:
                profile[kc] = round(random.uniform(0.05, 0.15), 2)
        
        # Assign hobbies to the student based on their ID
        student_hobbies = []
        num_hobbies = random.randint(1, 3)  # Each student gets 1-3 hobbies
        
        if 'alice' in student_id.lower():
            # Alice likes sports and music
            student_hobbies = ["sports", "music"]
            if num_hobbies == 3:
                student_hobbies.append(random.choice(["art", "books"]))
        
        elif 'bob' in student_id.lower():
            # Bob likes gaming and technology
            student_hobbies = ["gaming", "technology"]
            if num_hobbies == 3:
                student_hobbies.append(random.choice(["science", "movies"]))
        
        elif 'charlie' in student_id.lower():
            # Charlie likes cooking and travel
            student_hobbies = ["cooking", "travel"]
            if num_hobbies == 3:
                student_hobbies.append(random.choice(["history", "nature"]))
        
        elif 'dana' in student_id.lower():
            # Dana likes art and nature
            student_hobbies = ["art", "nature"]
            if num_hobbies == 3:
                student_hobbies.append(random.choice(["books", "fashion"]))
        
        elif 'evan' in student_id.lower():
            # Evan likes space and science
            student_hobbies = ["space", "science"]
            if num_hobbies == 3:
                student_hobbies.append(random.choice(["technology", "gaming"]))
        
        else:
            # For any other student, assign random hobbies
            student_hobbies = random.sample(self.possible_hobbies, num_hobbies)
        
        # Add hobbies to the profile dictionary as a separate return value
        # We return a tuple of (mastery_profile, hobbies)
        
        return profile, student_hobbies
