# services/recommendation/models/llm_content_generator.py

import random

class LLM_Content_Generator:
    """
    Simulates a Large Language Model (LLM) that can generate educational content
    on demand. This is the generative component of our hybrid model.
    """
    def __init__(self, kcs_map):
        self.kcs_map = kcs_map
        print("\n--- Building Recommendation Model Component ---")
        print("Initialized LLM Content Generator (simulated).")

    def generate_exercise(self, target_kc_id, difficulty='medium', hobbies=None):
        """
        Generates a new, tailored problem focusing on a specific knowledge component.
        Can be personalized based on student hobbies.
        
        Args:
            target_kc_id (str): The knowledge component to focus on
            difficulty (str): 'easy', 'medium', or 'hard'
            hobbies (list): Student's hobbies for personalization
            
        Returns:
            dict: Generated exercise with metadata
        """
        kc_name = self.kcs_map.get(target_kc_id, "a specific concept")
        new_exercise_id = f"gen_ex_{random.randint(100, 999)}"
        
        # Basic problem template
        problem_text = (f"Write a Python function that demonstrates your understanding of {kc_name}. "
                        f"Ensure your code is clean and handles edge cases.")
        
        # Adjust difficulty
        if difficulty == 'easy':
            problem_text = (f"Write a simple Python program that uses {kc_name}. "
                           f"Focus on the basic syntax and concepts.")
        elif difficulty == 'hard':
            problem_text = (f"Create an advanced Python program that utilizes {kc_name} in a complex scenario. "
                           f"Ensure your solution is efficient and handles all edge cases.")
        
        # Personalize based on hobbies if available
        if hobbies and len(hobbies) > 0:
            hobby = random.choice(hobbies)
            
            if hobby == 'sports':
                context = "sports statistics"
            elif hobby == 'music':
                context = "music playlist management"
            elif hobby == 'gaming':
                context = "game character attributes"
            elif hobby == 'cooking':
                context = "recipe management"
            elif hobby == 'art':
                context = "digital art parameters"
            elif hobby == 'science':
                context = "scientific data analysis"
            elif hobby == 'space':
                context = "planetary data"
            elif hobby == 'history':
                context = "historical timeline events"
            elif hobby == 'travel':
                context = "travel itinerary planning"
            elif hobby == 'nature':
                context = "ecological survey data"
            elif hobby == 'technology':
                context = "tech gadget specifications"
            elif hobby == 'movies':
                context = "movie database information"
            elif hobby == 'animals':
                context = "animal classification data"
            elif hobby == 'books':
                context = "book library management"
            elif hobby == 'fashion':
                context = "clothing inventory"
            else:
                context = hobby
            
            problem_text = (f"Create a Python program related to {context} that demonstrates "
                          f"your understanding of {kc_name}. " + problem_text)
        
        print(f"\nLLM generated a new problem '{new_exercise_id}' for KC '{target_kc_id}' at {difficulty} difficulty.")
        
        # Return structured exercise data
        return {
            'exercise_id': new_exercise_id,
            'target_kc': target_kc_id,
            'kc_name': kc_name,
            'difficulty': difficulty,
            'problem_text': problem_text,
            'personalized': hobbies is not None and len(hobbies) > 0,
            'hobby_context': hobbies[0] if hobbies and len(hobbies) > 0 else None,
            'hints': [
                f"Remember the key syntax for {kc_name}.",
                f"Consider how to implement {kc_name} efficiently.",
                f"Don't forget to handle edge cases."
            ]
        }

    def generate_explanation(self, target_kc_id, difficulty='medium'):
        """
        Generates an explanation for a knowledge component.
        
        Args:
            target_kc_id (str): The knowledge component to explain
            difficulty (str): Complexity level of the explanation
            
        Returns:
            dict: Generated explanation with metadata
        """
        kc_name = self.kcs_map.get(target_kc_id, "this concept")
        
        # Base explanation template
        base_explanation = (f"Let's review '{kc_name}'. This is a fundamental concept in programming.")
        
        # Add difficulty-specific content
        if target_kc_id == "KC1" or "Variables" in kc_name:
            if difficulty == 'easy':
                detail = ("Variables are containers that store data. In Python, you create a variable by "
                         "assigning a value to it with the '=' operator. For example: x = 10 creates a "
                         "variable named 'x' with the value 10.")
            elif difficulty == 'medium':
                detail = ("Variables in Python are dynamically typed, meaning you don't need to declare "
                         "their type explicitly. They can be reassigned to different data types. Variables "
                         "have scope rules that determine where they can be accessed from.")
            else:  # hard
                detail = ("Python variables are references to objects in memory. Understanding variable "
                         "scope (local, global, nonlocal), namespace management, and object mutability "
                         "is crucial for advanced Python programming. Variables can also have attributes "
                         "that affect how they behave, such as with special dunder methods.")
                
        elif target_kc_id == "KC2" or "Data Types" in kc_name:
            if difficulty == 'easy':
                detail = ("Python has several built-in data types: integers, floats, strings, booleans, "
                         "lists, tuples, and dictionaries. Each type stores different kinds of data and "
                         "has its own methods and operations.")
            elif difficulty == 'medium':
                detail = ("Python's data types can be categorized as mutable (lists, dictionaries, sets) "
                         "or immutable (integers, floats, strings, tuples). Understanding this distinction "
                         "is important for how data is stored and manipulated in memory.")
            else:  # hard
                detail = ("Advanced Python programming involves understanding the implementation details "
                         "of data types, such as how dictionaries use hash tables, or how memory is allocated "
                         "for different types. Custom data types can be created using classes, and type "
                         "hints can improve code readability and maintainability.")
        
        elif target_kc_id == "KC6" or "For Loops" in kc_name:
            if difficulty == 'easy':
                detail = ("For loops in Python are used to iterate over a sequence like a list, tuple, or string. "
                         "The basic syntax is: for item in sequence: followed by indented code. The loop "
                         "executes once for each item in the sequence.")
            elif difficulty == 'medium':
                detail = ("Python's for loops are actually 'for-each' loops that iterate over iterables. "
                         "The range() function is commonly used to create sequences of numbers. List "
                         "comprehensions provide a concise way to create lists using for loops.")
            else:  # hard
                detail = ("Advanced for loop techniques include using enumerate() to get indices, zip() to "
                         "iterate over multiple sequences simultaneously, and itertools module for complex "
                         "iterations. Generator expressions can be used for memory-efficient iteration.")
        
        else:
            # Generic explanation for other knowledge components
            if difficulty == 'easy':
                detail = (f"{kc_name} is a basic programming concept that helps you organize and manipulate data. "
                         f"It's important to understand the syntax and basic usage.")
            elif difficulty == 'medium':
                detail = (f"{kc_name} has various applications and built-in methods. Understanding how it "
                         f"interacts with other concepts can improve your code organization and efficiency.")
            else:  # hard
                detail = (f"Advanced usage of {kc_name} involves understanding its implementation details, "
                         f"performance characteristics, and best practices for complex scenarios.")
                
        # Combine the base explanation with the detailed content
        explanation = f"{base_explanation}\n\n{detail}"
        
        print(f"LLM generated an explanation for KC '{target_kc_id}' at {difficulty} difficulty.")
        
        # Return structured explanation data
        return {
            'kc_id': target_kc_id,
            'kc_name': kc_name,
            'difficulty': difficulty,
            'explanation_text': explanation,
            'examples': [
                f"Example 1: Basic usage of {kc_name}",
                f"Example 2: Common pattern with {kc_name}",
                f"Example 3: Advanced technique with {kc_name}"
            ]
        }

    def generate_hint(self, target_kc_id, difficulty='medium'):
        """
        Generates a hint for a specific knowledge component.
        
        Args:
            target_kc_id (str): The knowledge component to provide a hint for
            difficulty (str): Complexity level of the hint
            
        Returns:
            dict: Generated hint with metadata
        """
        kc_name = self.kcs_map.get(target_kc_id, "this concept")
        
        # Generate hint based on KC and difficulty
        if target_kc_id == "KC1" or "Variables" in kc_name:
            if difficulty == 'easy':
                hint = f"Remember that variables store data. Try using the assignment operator (=) to create a variable."
            elif difficulty == 'medium':
                hint = f"Consider the scope of your variables. Are they accessible where you're trying to use them?"
            else:  # hard
                hint = f"Think about variable mutability and memory references. Are you modifying the original object or a copy?"
                
        elif target_kc_id == "KC2" or "Data Types" in kc_name:
            if difficulty == 'easy':
                hint = f"Check the data type of your variables using type(). Make sure you're using the appropriate operations."
            elif difficulty == 'medium':
                hint = f"Consider whether you need a mutable or immutable data type for this task. How will it affect your code?"
            else:  # hard
                hint = f"Think about the memory and performance implications of your chosen data structures. Could a different type be more efficient?"
        
        elif target_kc_id == "KC6" or "For Loops" in kc_name:
            if difficulty == 'easy':
                hint = f"A for loop iterates over a sequence. Make sure you're using the correct sequence to loop through."
            elif difficulty == 'medium':
                hint = f"Consider using range(), enumerate(), or zip() to handle your iteration needs more efficiently."
            else:  # hard
                hint = f"Could you use list comprehensions or generator expressions instead of a traditional for loop? Think about readability vs. conciseness."
        
        else:
            # Generic hint for other knowledge components
            if difficulty == 'easy':
                hint = f"Review the basic syntax for {kc_name}. Make sure you're following the correct format."
            elif difficulty == 'medium':
                hint = f"Think about how {kc_name} interacts with other parts of your code. Are there any conflicts?"
            else:  # hard
                hint = f"Consider the edge cases and performance implications of your approach to {kc_name}."
        
        print(f"LLM generated a hint for KC '{target_kc_id}' at {difficulty} difficulty.")
        
        # Return structured hint data
        return {
            'kc_id': target_kc_id,
            'kc_name': kc_name,
            'difficulty': difficulty,
            'hint_text': hint,
            'follow_up_hints': [
                "Still stuck? Try breaking down the problem into smaller steps.",
                "Consider reviewing the documentation or examples for this concept.",
                "Try working through a simpler example first to understand the pattern."
            ]
        }

    def generate_new_problem(self, target_kc_id):
        """
        Legacy method for backward compatibility.
        Now uses generate_exercise internally.
        """
        exercise = self.generate_exercise(target_kc_id)
        return exercise['exercise_id'], exercise['problem_text']
