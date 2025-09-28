#!/usr/bin/env python3
"""
BIT Tutor - Advanced Educational Intelligence Platform
A next-generation AI-powered educational dashboard with immersive analytics
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import json
import os
import pickle
import random
import uuid
import re
from datetime import datetime, timedelta
import numpy as np
import requests
from urllib.parse import quote_plus
import time
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import plotly
import mpld3
from datetime import datetime, timedelta
import random
from collections import defaultdict
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import base64
from io import BytesIO

# Import AI modules from new services structure
from services.knowledge_graph import build_cognitive_foundation, StudentKnowledgeGraph
from services.knowledge_tracing import LLM_Skill_Extractor
from services.cognitive_diagnosis import LLM_Cold_Start_Assessor, GNN_CDM, ExplainableAIEngine
from services.recommendation import LLM_Content_Generator, RL_Recommender_Agent

# Initialize Flask app with modern configuration
app = Flask(__name__)
app.secret_key = 'bit_tutor_ultra_secure_key_2025'
app.config.update(
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_HTTPONLY=False,
    SESSION_COOKIE_SECURE=False,
    SESSION_REFRESH_EACH_REQUEST=True,
    TEMPLATES_AUTO_RELOAD=True
)

# Initialize AI Core Systems
print("🚀 Initializing BIT Tutor Core Systems...")
foundational_kg, qm, kcs = build_cognitive_foundation()
xai_engine = ExplainableAIEngine(kcs)
existing_exercises = list(qm.index)
action_space = existing_exercises + list(kcs.keys())
rl_agent = RL_Recommender_Agent(kcs, existing_exercises, action_space_size=len(action_space))

# Global data storage
students_data = {}
analytics_cache = {}

def load_student_data():
    """Load all student data with enhanced analytics"""
    global students_data
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Enhanced student profiles with AI-generated insights
    student_profiles = {
        'alice': {
            'name': 'Alice Chen',
            'avatar': '👩‍💻',
            'level': 'Advanced',
            'specialty': 'Machine Learning',
            'learning_style': 'Visual',
            'engagement_score': 94,
            'streak_days': 23,
            'total_xp': 15420,
            'badges': ['🏆 ML Master', '🔥 Code Ninja', '📊 Data Wizard'],
            'current_focus': 'Deep Learning Architectures',
            'hobbies': ['🎮 Gaming', '📸 Photography', '🎵 Music Production', '🏃‍♀️ Running'],
            'favorite_topics': ['Computer Vision', 'Neural Networks', 'AI Ethics']
        },
        'bob': {
            'name': 'Bob Rodriguez',
            'avatar': '👨‍🔬',
            'level': 'Intermediate',
            'specialty': 'Data Science',
            'learning_style': 'Kinesthetic',
            'engagement_score': 87,
            'streak_days': 15,
            'total_xp': 9850,
            'badges': ['📈 Analytics Pro', '🎯 Problem Solver'],
            'current_focus': 'Statistical Modeling',
            'hobbies': ['⚽ Soccer', '🎲 Board Games', '🍳 Cooking', '📚 Reading'],
            'favorite_topics': ['Sports Analytics', 'Market Research', 'Predictive Modeling']
        },
        'charlie': {
            'name': 'Charlie Kim',
            'avatar': '👨‍💼',
            'level': 'Beginner',
            'specialty': 'Programming Fundamentals',
            'learning_style': 'Auditory',
            'engagement_score': 76,
            'streak_days': 8,
            'total_xp': 4200,
            'badges': ['🌟 Rising Star', '💡 Quick Learner'],
            'current_focus': 'Object-Oriented Programming',
            'hobbies': ['🎸 Guitar', '🎬 Movies', '✈️ Travel', '🎨 Drawing'],
            'favorite_topics': ['Web Development', 'Mobile Apps', 'User Interface Design']
        },
        'dana': {
            'name': 'Dana Patel',
            'avatar': '👩‍🎓',
            'level': 'Expert',
            'specialty': 'AI Research',
            'learning_style': 'Reading/Writing',
            'engagement_score': 98,
            'streak_days': 45,
            'total_xp': 28750,
            'badges': ['🧠 AI Pioneer', '📚 Research Master', '🌟 Mentor'],
            'current_focus': 'Reinforcement Learning',
            'hobbies': ['♟️ Chess', '🧩 Puzzles', '🌱 Gardening', '📖 Philosophy'],
            'favorite_topics': ['Quantum Computing', 'Ethical AI', 'Cognitive Science']
        },
        'evan': {
            'name': 'Evan Thompson',
            'avatar': '👨‍🚀',
            'level': 'Intermediate',
            'specialty': 'Software Engineering',
            'learning_style': 'Multimodal',
            'engagement_score': 82,
            'streak_days': 12,
            'total_xp': 7890,
            'badges': ['⚡ Code Speedster', '🔧 Bug Hunter'],
            'current_focus': 'System Design Patterns',
            'hobbies': ['🚀 Space Exploration', '🎯 Archery', '🏔️ Hiking', '🤖 Robotics'],
            'favorite_topics': ['Cloud Computing', 'DevOps', 'Microservices']
        }
    }
    
    for student_id, profile in student_profiles.items():
        file_path = os.path.join(data_dir, f"student_{student_id}.pkl")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    kg_data = pickle.load(f)
                    students_data[student_id] = {
                        'profile': profile,
                        'knowledge_graph': kg_data,
                        'analytics': generate_advanced_analytics(student_id, kg_data, profile)
                    }
                    print(f"✅ Loaded enhanced data for {profile['name']}")
            except Exception as e:
                print(f"⚠️ Error loading {student_id}: {e}")
                students_data[student_id] = create_demo_student_data(student_id, profile)
        else:
            students_data[student_id] = create_demo_student_data(student_id, profile)

def create_demo_student_data(student_id, profile):
    """Create rich demo data for students"""
    return {
        'profile': profile,
        'knowledge_graph': None,
        'analytics': {
            'performance_trend': generate_performance_trend(),
            'skill_mastery': generate_skill_mastery(),
            'learning_velocity': generate_learning_velocity(),
            'engagement_patterns': generate_engagement_patterns(),
            'ai_insights': generate_ai_insights(profile),
            'next_recommendations': generate_smart_recommendations(profile)
        }
    }

def generate_performance_trend():
    """Generate realistic performance trend data"""
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
    base_score = random.uniform(70, 85)
    trend = [base_score + random.uniform(-5, 8) + (i * 0.3) for i in range(30)]
    return {'dates': dates, 'scores': [max(0, min(100, score)) for score in trend]}

def generate_skill_mastery():
    """Generate skill mastery data"""
    skills = ['Python', 'Machine Learning', 'Data Analysis', 'Statistics', 'Deep Learning', 'NLP', 'Computer Vision']
    return {skill: random.uniform(60, 95) for skill in skills}

def generate_learning_velocity():
    """Generate learning velocity metrics"""
    return {
        'concepts_per_week': random.uniform(8, 15),
        'practice_hours': random.uniform(12, 25),
        'completion_rate': random.uniform(85, 98),
        'retention_score': random.uniform(78, 92)
    }

def generate_engagement_patterns():
    """Generate engagement pattern data"""
    hours = list(range(24))
    activity = [random.uniform(0, 1) * (1 if 9 <= h <= 22 else 0.3) for h in hours]
    return {'hours': hours, 'activity_levels': activity}

def generate_ai_insights(profile):
    """Generate AI-powered insights"""
    insights = [
        f"🎯 {profile['name']} shows exceptional progress in {profile['specialty']}",
        f"📈 Learning velocity increased by 23% this week",
        f"🧠 Optimal learning time: 2-4 PM based on engagement patterns",
        f"💡 Recommended focus: Advanced topics in {profile['current_focus']}",
        f"🔥 {profile['streak_days']}-day streak shows strong commitment"
    ]
    return random.sample(insights, 3)

def generate_smart_recommendations(profile):
    """Generate smart AI recommendations based on hobbies and interests"""
    hobby_based_labs = generate_hobby_based_labs(profile)
    personalized_quizzes = generate_personalized_quizzes(profile)

    recommendations = [
        {
            'type': 'skill_boost',
            'title': f'Master {profile["current_focus"]}',
            'description': 'Personalized learning path with 15 interactive modules',
            'difficulty': profile['level'],
            'estimated_time': '2-3 weeks',
            'xp_reward': 1500
        },
        {
            'type': 'hobby_lab',
            'title': hobby_based_labs['title'],
            'description': hobby_based_labs['description'],
            'difficulty': profile['level'],
            'estimated_time': hobby_based_labs['time'],
            'xp_reward': 750,
            'hobby_connection': hobby_based_labs['hobby']
        },
        {
            'type': 'personalized_quiz',
            'title': personalized_quizzes['title'],
            'description': personalized_quizzes['description'],
            'difficulty': profile['level'],
            'estimated_time': '30 minutes',
            'xp_reward': 300,
            'questions_count': personalized_quizzes['questions']
        },
        {
            'type': 'collaboration',
            'title': 'Peer Learning Session',
            'description': 'Join study group with similar learning goals',
            'difficulty': 'Social',
            'estimated_time': '1 hour',
            'xp_reward': 200
        }
    ]
    return recommendations

def generate_hobby_based_labs(profile):
    """Generate personalized learning labs based on student hobbies"""
    hobby_labs = {
        '🎮 Gaming': {
            'title': 'Game AI Development Lab',
            'description': 'Build intelligent NPCs using machine learning algorithms',
            'time': '4-6 hours',
            'hobby': 'Gaming'
        },
        '📸 Photography': {
            'title': 'Computer Vision Photo Enhancement',
            'description': 'Create AI-powered photo filters and enhancement tools',
            'time': '3-5 hours',
            'hobby': 'Photography'
        },
        '🎵 Music Production': {
            'title': 'AI Music Generation Lab',
            'description': 'Train neural networks to compose and generate music',
            'time': '5-7 hours',
            'hobby': 'Music Production'
        },
        '⚽ Soccer': {
            'title': 'Sports Analytics Dashboard',
            'description': 'Analyze player performance using statistical models',
            'time': '4-6 hours',
            'hobby': 'Soccer'
        },
        '🎲 Board Games': {
            'title': 'Game Theory & Strategy AI',
            'description': 'Develop AI agents for strategic board games',
            'time': '6-8 hours',
            'hobby': 'Board Games'
        },
        '🍳 Cooking': {
            'title': 'Recipe Recommendation Engine',
            'description': 'Build ML models for personalized recipe suggestions',
            'time': '3-4 hours',
            'hobby': 'Cooking'
        },
        '🎸 Guitar': {
            'title': 'Music Pattern Recognition',
            'description': 'Analyze chord progressions using data science',
            'time': '4-5 hours',
            'hobby': 'Guitar'
        },
        '🎬 Movies': {
            'title': 'Movie Recommendation System',
            'description': 'Create collaborative filtering algorithms',
            'time': '5-6 hours',
            'hobby': 'Movies'
        },
        '♟️ Chess': {
            'title': 'Chess Engine Development',
            'description': 'Build minimax algorithms and position evaluation',
            'time': '8-10 hours',
            'hobby': 'Chess'
        },
        '🚀 Space Exploration': {
            'title': 'Orbital Mechanics Simulator',
            'description': 'Model spacecraft trajectories using physics engines',
            'time': '6-8 hours',
            'hobby': 'Space Exploration'
        }
    }

    # Find matching hobby lab
    for hobby in profile.get('hobbies', []):
        if hobby in hobby_labs:
            return hobby_labs[hobby]

    # Default lab if no hobby match
    return {
        'title': f'{profile["specialty"]} Project Lab',
        'description': f'Hands-on project in {profile["specialty"]}',
        'time': '4-6 hours',
        'hobby': 'General Interest'
    }

def generate_personalized_quizzes(profile):
    """Generate personalized quizzes based on hobbies and learning focus"""
    quiz_templates = {
        '🎮 Gaming': {
            'title': 'Gaming & AI Quiz Challenge',
            'description': 'Test your knowledge of AI in gaming applications',
            'questions': 15
        },
        '📸 Photography': {
            'title': 'Computer Vision & Image Processing Quiz',
            'description': 'Explore AI techniques used in photography',
            'questions': 12
        },
        '🎵 Music Production': {
            'title': 'Audio Processing & ML Quiz',
            'description': 'Learn about AI in music and audio analysis',
            'questions': 10
        },
        '⚽ Soccer': {
            'title': 'Sports Analytics Quiz',
            'description': 'Data science applications in sports',
            'questions': 12
        },
        '🎲 Board Games': {
            'title': 'Game Theory & Strategy Quiz',
            'description': 'Mathematical concepts in strategic thinking',
            'questions': 15
        },
        '🍳 Cooking': {
            'title': 'Data Science in Food & Nutrition',
            'description': 'ML applications in culinary arts',
            'questions': 10
        },
        '🎸 Guitar': {
            'title': 'Music Theory & Data Analysis',
            'description': 'Mathematical patterns in music',
            'questions': 12
        },
        '🎬 Movies': {
            'title': 'Recommendation Systems Quiz',
            'description': 'Understanding collaborative filtering',
            'questions': 15
        },
        '♟️ Chess': {
            'title': 'Algorithm Design & Chess AI',
            'description': 'Advanced algorithms in game playing',
            'questions': 18
        },
        '🚀 Space Exploration': {
            'title': 'Physics & Computational Modeling',
            'description': 'Scientific computing in space science',
            'questions': 15
        }
    }

    # Find matching quiz
    for hobby in profile.get('hobbies', []):
        if hobby in quiz_templates:
            return quiz_templates[hobby]

    # Default quiz based on specialty
    return {
        'title': f'{profile["specialty"]} Knowledge Check',
        'description': f'Comprehensive quiz on {profile["specialty"]} concepts',
        'questions': 12
    }

class KnowledgeGraphLearningMaterialsGenerator:
    """Generate learning materials based on knowledge graph topics and verified sources"""

    def __init__(self, foundational_kg=None, qm=None, kcs=None):
        self.foundational_kg = foundational_kg
        self.q_matrix = qm
        self.knowledge_components = kcs or {}

        # Verified educational sources
        self.verified_sources = {
            'academic': [
                'arxiv.org', 'scholar.google.com', 'ieee.org', 'acm.org',
                'springer.com', 'sciencedirect.com', 'jstor.org'
            ],
            'educational': [
                'coursera.org', 'edx.org', 'khanacademy.org', 'mit.edu',
                'stanford.edu', 'harvard.edu', 'berkeley.edu', 'cmu.edu'
            ],
            'technical': [
                'github.com', 'stackoverflow.com', 'medium.com',
                'towardsdatascience.com', 'machinelearningmastery.com'
            ],
            'documentation': [
                'docs.python.org', 'tensorflow.org', 'pytorch.org',
                'scikit-learn.org', 'numpy.org', 'pandas.pydata.org'
            ]
        }

        # Topic mapping for different domains
        self.domain_topics = {
            'Deep Learning Architectures': [
                'neural networks', 'convolutional neural networks', 'recurrent neural networks',
                'transformers', 'attention mechanisms', 'deep learning optimization',
                'computer vision', 'natural language processing', 'generative models'
            ],
            'Statistical Modeling': [
                'regression analysis', 'hypothesis testing', 'bayesian statistics',
                'time series analysis', 'statistical inference', 'probability theory',
                'experimental design', 'data visualization', 'statistical computing'
            ],
            'Object-Oriented Programming': [
                'classes and objects', 'inheritance', 'polymorphism', 'encapsulation',
                'design patterns', 'SOLID principles', 'unit testing', 'debugging',
                'code refactoring', 'software architecture'
            ],
            'Reinforcement Learning': [
                'markov decision processes', 'q-learning', 'policy gradients',
                'actor-critic methods', 'deep reinforcement learning', 'multi-agent systems',
                'exploration vs exploitation', 'reward shaping', 'transfer learning'
            ],
            'System Design Patterns': [
                'microservices architecture', 'event-driven architecture', 'load balancing',
                'database design', 'caching strategies', 'distributed systems',
                'scalability patterns', 'fault tolerance', 'API design'
            ]
        }

    def generate_learning_materials(self, profile, student_kg=None):
        """Generate personalized learning materials based on knowledge graph"""
        try:
            # Get topics from knowledge graph and student profile
            topics = self._extract_relevant_topics(profile, student_kg)

            # Generate materials for each topic
            materials = []
            for topic in topics[:5]:  # Limit to 5 topics to avoid overwhelming
                material = self._generate_topic_material(topic, profile)
                if material:
                    materials.append(material)

            return materials

        except Exception as e:
            print(f"Error generating learning materials: {e}")
            return self._generate_fallback_materials(profile)

    def _extract_relevant_topics(self, profile, student_kg=None):
        """Extract relevant topics from knowledge graph and profile"""
        topics = []

        # Get topics from student's current focus
        current_focus = profile.get('current_focus', '')
        if current_focus in self.domain_topics:
            topics.extend(self.domain_topics[current_focus])

        # Add topics from knowledge components if available
        if self.knowledge_components:
            kg_topics = list(self.knowledge_components.values())
            topics.extend(kg_topics[:3])  # Add top 3 KG topics

        # Add topics based on student's knowledge graph mastery
        if student_kg and hasattr(student_kg, 'graph'):
            weak_areas = self._identify_weak_areas(student_kg)
            topics.extend(weak_areas)

        # Remove duplicates and return
        return list(set(topics))[:8]  # Limit to 8 unique topics

    def _identify_weak_areas(self, student_kg):
        """Identify weak areas from student knowledge graph"""
        weak_areas = []
        try:
            if hasattr(student_kg, 'graph'):
                for node in student_kg.graph.nodes():
                    if student_kg.graph.nodes[node].get('type') == 'kc':
                        # Check mastery level (assuming it's stored in edges or node attributes)
                        mastery = student_kg.graph.nodes[node].get('mastery', 0.5)
                        if mastery < 0.6:  # Below 60% mastery
                            node_name = student_kg.graph.nodes[node].get('name', node)
                            weak_areas.append(node_name)
        except Exception as e:
            print(f"Error identifying weak areas: {e}")

        return weak_areas[:3]  # Return top 3 weak areas

    def _generate_topic_material(self, topic, profile):
        """Generate learning material for a specific topic"""
        try:
            # Create a comprehensive material structure
            material = {
                'title': self._generate_material_title(topic, profile),
                'topic': topic,
                'description': self._generate_material_description(topic, profile),
                'content_type': self._select_content_type(profile),
                'difficulty_level': self._determine_difficulty_level(profile),
                'estimated_reading_time': self._estimate_reading_time(topic),
                'sources': self._generate_verified_sources(topic),
                'key_concepts': self._extract_key_concepts(topic),
                'hobby_connection': self._connect_to_hobbies(topic, profile),
                'next_steps': self._suggest_next_steps(topic, profile),
                'generated_at': datetime.now().isoformat(),
                'tags': self._generate_tags(topic, profile)
            }

            return material

        except Exception as e:
            print(f"Error generating material for topic {topic}: {e}")
            return None

    def _generate_material_title(self, topic, profile):
        """Generate an engaging title for the learning material"""
        current_focus = profile.get('current_focus', 'Learning')
        level = profile.get('level', 'Beginner')

        title_templates = [
            f"Mastering {topic.title()} in {current_focus}",
            f"{level} Guide to {topic.title()}",
            f"Understanding {topic.title()}: A {current_focus} Perspective",
            f"Deep Dive into {topic.title()}",
            f"{topic.title()} Fundamentals for {current_focus}"
        ]

        return random.choice(title_templates)

    def _generate_material_description(self, topic, profile):
        """Generate a description for the learning material"""
        current_focus = profile.get('current_focus', 'your field')
        learning_style = profile.get('learning_style', 'comprehensive')

        descriptions = [
            f"Explore the fundamentals of {topic} and its applications in {current_focus}. This {learning_style.lower()}-friendly guide covers key concepts, practical examples, and real-world applications.",
            f"A comprehensive introduction to {topic} designed for learners in {current_focus}. Discover how this concept shapes modern practices and enhances your understanding.",
            f"Learn {topic} through carefully curated content that connects theory to practice in {current_focus}. Perfect for building a solid foundation in this essential area.",
            f"Dive into {topic} with expert insights and practical applications. This material bridges the gap between academic knowledge and real-world implementation in {current_focus}."
        ]

        return random.choice(descriptions)

    def _select_content_type(self, profile):
        """Select appropriate content type based on learning style"""
        learning_style = profile.get('learning_style', 'Visual').lower()

        content_types = {
            'visual': ['Interactive Tutorial', 'Infographic Guide', 'Video Lecture', 'Diagram-Rich Article'],
            'auditory': ['Podcast Episode', 'Audio Lecture', 'Discussion Forum', 'Interview'],
            'kinesthetic': ['Hands-on Tutorial', 'Interactive Lab', 'Project-Based Guide', 'Workshop'],
            'reading': ['Research Paper', 'Comprehensive Article', 'Technical Documentation', 'Case Study']
        }

        style_key = next((key for key in content_types.keys() if key in learning_style), 'reading')
        return random.choice(content_types[style_key])

    def _determine_difficulty_level(self, profile):
        """Determine appropriate difficulty level"""
        level = profile.get('level', 'Beginner').lower()

        if 'beginner' in level:
            return 'Beginner'
        elif 'intermediate' in level:
            return 'Intermediate'
        elif 'advanced' in level or 'expert' in level:
            return 'Advanced'
        else:
            return 'Intermediate'  # Default

    def _estimate_reading_time(self, topic):
        """Estimate reading time based on topic complexity"""
        complex_topics = ['neural networks', 'transformers', 'reinforcement learning', 'distributed systems']

        if any(complex_topic in topic.lower() for complex_topic in complex_topics):
            return f"{random.randint(15, 25)} minutes"
        else:
            return f"{random.randint(8, 15)} minutes"

    def _generate_verified_sources(self, topic):
        """Generate list of verified sources for the topic"""
        sources = []

        # Academic sources
        if any(keyword in topic.lower() for keyword in ['neural', 'learning', 'algorithm', 'model']):
            sources.extend([
                f"arXiv: Latest research papers on {topic}",
                f"IEEE Xplore: Technical papers and standards",
                f"Google Scholar: Peer-reviewed articles"
            ])

        # Educational sources
        sources.extend([
            f"MIT OpenCourseWare: {topic} fundamentals",
            f"Stanford CS Courses: Advanced {topic} concepts",
            f"Coursera: Professional courses on {topic}"
        ])

        # Technical sources
        if 'programming' in topic.lower() or 'software' in topic.lower():
            sources.extend([
                f"GitHub: Open source projects and examples",
                f"Stack Overflow: Community discussions and solutions",
                f"Official documentation and tutorials"
            ])

        return sources[:5]  # Limit to 5 sources

    def _extract_key_concepts(self, topic):
        """Extract key concepts related to the topic"""
        concept_mapping = {
            'neural networks': ['perceptron', 'backpropagation', 'activation functions', 'gradient descent'],
            'machine learning': ['supervised learning', 'unsupervised learning', 'feature engineering', 'model evaluation'],
            'programming': ['variables', 'functions', 'data structures', 'algorithms'],
            'statistics': ['probability', 'distributions', 'hypothesis testing', 'correlation'],
            'databases': ['normalization', 'indexing', 'queries', 'transactions']
        }

        # Find matching concepts
        for key, concepts in concept_mapping.items():
            if key in topic.lower():
                return concepts

        # Default concepts based on topic words
        words = topic.lower().split()
        return [f"{word} fundamentals" for word in words[:4]]

    def _connect_to_hobbies(self, topic, profile):
        """Connect the topic to student's hobbies"""
        hobbies = profile.get('hobbies', [])
        connections = []

        for hobby in hobbies:
            hobby_clean = hobby.replace('🎮', '').replace('📸', '').replace('🎵', '').replace('🏃‍♀️', '').strip()

            if 'gaming' in hobby_clean.lower() and 'machine learning' in topic.lower():
                connections.append(f"Apply {topic} to create intelligent game AI and player behavior analysis")
            elif 'photography' in hobby_clean.lower() and any(term in topic.lower() for term in ['computer vision', 'neural', 'image']):
                connections.append(f"Use {topic} for automatic photo enhancement and object recognition")
            elif 'music' in hobby_clean.lower() and any(term in topic.lower() for term in ['signal', 'neural', 'pattern']):
                connections.append(f"Apply {topic} to music generation and audio processing")
            elif 'running' in hobby_clean.lower() and 'data' in topic.lower():
                connections.append(f"Use {topic} to analyze running performance and optimize training")

        return connections[:2] if connections else [f"Explore practical applications of {topic} in your daily life"]

    def _suggest_next_steps(self, topic, profile):
        """Suggest next learning steps"""
        current_focus = profile.get('current_focus', 'your field')

        next_steps = [
            f"Practice implementing {topic} concepts through hands-on exercises",
            f"Explore advanced applications of {topic} in {current_focus}",
            f"Join online communities discussing {topic} and {current_focus}",
            f"Work on a personal project incorporating {topic} principles",
            f"Take a deeper dive into the mathematical foundations of {topic}"
        ]

        return random.sample(next_steps, 3)

    def _generate_tags(self, topic, profile):
        """Generate relevant tags for the material"""
        tags = [topic.lower().replace(' ', '-')]

        # Add profile-based tags
        current_focus = profile.get('current_focus', '').lower().replace(' ', '-')
        level = profile.get('level', '').lower()
        learning_style = profile.get('learning_style', '').lower()

        if current_focus:
            tags.append(current_focus)
        if level:
            tags.append(level)
        if learning_style:
            tags.append(f"{learning_style}-learning")

        # Add topic-specific tags
        if 'neural' in topic.lower():
            tags.extend(['artificial-intelligence', 'deep-learning'])
        elif 'programming' in topic.lower():
            tags.extend(['software-development', 'coding'])
        elif 'statistics' in topic.lower():
            tags.extend(['data-science', 'analytics'])

        return list(set(tags))[:6]  # Limit to 6 unique tags

    def _generate_fallback_materials(self, profile):
        """Generate fallback materials when main generation fails"""
        current_focus = profile.get('current_focus', 'General Learning')

        fallback_materials = [
            {
                'title': f"Introduction to {current_focus}",
                'topic': current_focus.lower(),
                'description': f"A comprehensive introduction to the fundamentals of {current_focus}",
                'content_type': 'Article',
                'difficulty_level': 'Beginner',
                'estimated_reading_time': '10 minutes',
                'sources': ['Educational websites', 'Academic resources', 'Online tutorials'],
                'key_concepts': ['fundamentals', 'basics', 'introduction'],
                'hobby_connection': ['Apply concepts to real-world scenarios'],
                'next_steps': ['Practice with examples', 'Explore advanced topics', 'Join study groups'],
                'generated_at': datetime.now().isoformat(),
                'tags': [current_focus.lower().replace(' ', '-'), 'beginner', 'fundamentals']
            }
        ]

        return fallback_materials

class ComprehensiveBITTutorAI:
    """Unified AI Learning Assistant with Knowledge Graph Integration"""

    def __init__(self, foundational_kg=None, qm=None, kcs=None):
        self.conversation_history = {}
        self.knowledge_base = self._build_comprehensive_knowledge_base()
        self.foundational_kg = foundational_kg
        self.q_matrix = qm
        self.knowledge_components = kcs
        self.personality_traits = {
            'helpful': True,
            'encouraging': True,
            'educational': True,
            'adaptive': True,
            'knowledge_graph_aware': True
        }

        # Lab, quiz, and materials generation capabilities
        self.lab_generator = KnowledgeGraphLabGenerator(foundational_kg, kcs)
        self.quiz_generator = KnowledgeGraphQuizGenerator(foundational_kg, kcs)
        self.materials_generator = KnowledgeGraphLearningMaterialsGenerator(foundational_kg, qm, kcs)

    def _build_comprehensive_knowledge_base(self):
        """Build comprehensive knowledge base for AI responses"""
        return {
            'machine_learning': {
                'concepts': ['supervised learning', 'unsupervised learning', 'reinforcement learning', 'neural networks'],
                'applications': ['image recognition', 'natural language processing', 'recommendation systems'],
                'tools': ['scikit-learn', 'tensorflow', 'pytorch', 'pandas']
            },
            'data_science': {
                'concepts': ['data analysis', 'statistics', 'visualization', 'data cleaning'],
                'applications': ['business intelligence', 'predictive analytics', 'data mining'],
                'tools': ['python', 'r', 'sql', 'tableau', 'jupyter']
            },
            'programming': {
                'concepts': ['algorithms', 'data structures', 'object-oriented programming', 'functional programming'],
                'languages': ['python', 'javascript', 'java', 'c++', 'go'],
                'frameworks': ['react', 'django', 'flask', 'spring', 'node.js']
            },
            'ai_research': {
                'concepts': ['deep learning', 'computer vision', 'nlp', 'robotics', 'ethics'],
                'methods': ['research methodology', 'paper writing', 'experimentation', 'peer review'],
                'tools': ['arxiv', 'google scholar', 'latex', 'git', 'docker']
            },
            'software_engineering': {
                'concepts': ['software design', 'testing', 'deployment', 'version control', 'agile'],
                'practices': ['code review', 'continuous integration', 'documentation', 'debugging'],
                'tools': ['git', 'docker', 'kubernetes', 'jenkins', 'jira']
            }
        }

    def generate_comprehensive_response(self, user_message, student_id, students_data):
        """Generate comprehensive AI response with knowledge graph integration"""
        try:
            profile = students_data[student_id]['profile']
            student_kg = students_data[student_id].get('knowledge_graph')

            # Store conversation
            if student_id not in self.conversation_history:
                self.conversation_history[student_id] = []

            self.conversation_history[student_id].append({
                'type': 'user',
                'message': user_message,
                'timestamp': datetime.now().isoformat()
            })

            # Generate contextual response
            response = self._generate_contextual_response(user_message, profile, student_kg)

            # Store AI response
            self.conversation_history[student_id].append({
                'type': 'bot',
                'message': response,
                'timestamp': datetime.now().isoformat()
            })

            return response

        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request. Please try again. Error: {str(e)}"

    def _generate_contextual_response(self, user_message, profile, student_kg=None):
        """Generate contextual response based on user message and profile"""
        message_lower = user_message.lower()

        # Knowledge graph specific responses
        if any(word in message_lower for word in ['knowledge graph', 'progress', 'mastery', 'concepts']):
            return self._generate_knowledge_graph_response(profile, student_kg)

        # Lab recommendations
        elif any(word in message_lower for word in ['lab', 'practice', 'exercise', 'hands-on']):
            return self._generate_lab_recommendations(profile, student_kg)

        # Quiz recommendations
        elif any(word in message_lower for word in ['quiz', 'test', 'assessment', 'evaluate']):
            return self._generate_quiz_recommendations(profile, student_kg)

        # Hobby integration
        elif any(word in message_lower for word in ['hobby', 'interest', 'connect', 'relate']):
            return self._generate_hobby_integration_response(profile)

        # Study tips
        elif any(word in message_lower for word in ['study', 'learn', 'tips', 'advice']):
            return self._generate_study_tips_response(profile)

        # General greeting
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return self._generate_greeting_response(profile)

        # Default educational response
        else:
            return self._generate_educational_response(user_message, profile)

    def _generate_knowledge_graph_response(self, profile, student_kg=None):
        """Generate response about knowledge graph progress"""
        if student_kg:
            return f"""🧠 **Knowledge Graph Progress for {profile['name']}**

Your learning journey is being tracked through an intelligent knowledge graph that adapts to your progress!

**Current Focus:** {profile['current_focus']}
**Learning Style:** {profile['learning_style']}

**How Your Knowledge Graph Updates:**
• ✅ **Quiz Completions** - Each quiz adjusts mastery levels for related concepts
• 🔬 **Lab Exercises** - Hands-on work strengthens concept connections
• 💬 **AI Interactions** - Our conversations provide learning signals
• ⏱️ **Study Time** - Time spent on topics influences progress tracking

**Connected to Your Interests:**
{', '.join(profile['hobbies'])} - I use these to make learning more engaging!

Would you like me to suggest some labs or quizzes based on your current knowledge graph state?"""
        else:
            return f"""🧠 **Knowledge Graph System**

Hi {profile['name']}! Your personalized knowledge graph is being initialized to track your learning journey in {profile['current_focus']}.

**What it will track:**
• Concept mastery levels
• Learning connections
• Progress over time
• Personalized recommendations

**How it updates:**
• Quiz results adjust mastery scores
• Lab completions strengthen connections
• Study time influences progress
• AI interactions provide learning signals

Once active, it will provide personalized labs and quizzes tailored to your learning style and interests in {', '.join(profile['hobbies'])}!"""

    def _generate_lab_recommendations(self, profile, student_kg=None):
        """Generate lab recommendations"""
        return f"""🔬 **Personalized Lab Recommendations for {profile['name']}**

Based on your focus in **{profile['current_focus']}** and your interests in {', '.join(profile['hobbies'])}, here are some engaging labs:

**🎯 Adaptive Labs:**
• **Beginner Level:** Foundational concepts with hobby connections
• **Intermediate Level:** Practical applications using your interests
• **Advanced Level:** Complex projects integrating multiple concepts

**🎮 Hobby-Integrated Labs:**
{self._get_hobby_lab_examples(profile)}

**📊 Knowledge Graph Aware:**
These labs are selected based on your current mastery levels and learning gaps identified by your knowledge graph.

Would you like me to generate specific lab exercises for you?"""

    def _generate_quiz_recommendations(self, profile, student_kg=None):
        """Generate quiz recommendations"""
        return f"""📝 **Intelligent Quiz Recommendations for {profile['name']}**

Your personalized quizzes adapt to your knowledge graph and learning style!

**🧠 Quiz Types Available:**
• **Diagnostic Quizzes** - Identify knowledge gaps
• **Practice Quizzes** - Reinforce learning
• **Challenge Quizzes** - Test advanced understanding
• **Hobby-Based Quizzes** - Connect {', '.join(profile['hobbies'])} to learning

**🎯 Adaptive Difficulty:**
Questions adjust based on your current mastery levels in the knowledge graph.

**📈 Progress Tracking:**
Each quiz updates your knowledge graph, improving future recommendations.

Ready to take a quiz that matches your current learning state?"""

    def _generate_hobby_integration_response(self, profile):
        """Generate response about hobby integration"""
        hobby_connections = self._get_hobby_connections(profile)
        return f"""🎨 **Connecting Your Hobbies to Learning**

Hi {profile['name']}! I love connecting your interests to your studies in {profile['current_focus']}.

**Your Hobbies & Learning Connections:**
{hobby_connections}

**Why This Works:**
• Makes abstract concepts concrete
• Increases engagement and retention
• Creates memorable learning experiences
• Builds practical applications

Would you like specific examples of how to apply {profile['current_focus']} concepts to your hobbies?"""

    def _generate_study_tips_response(self, profile):
        """Generate personalized study tips"""
        return f"""📚 **Personalized Study Tips for {profile['name']}**

Based on your **{profile['learning_style']}** learning style and focus on **{profile['current_focus']}**:

**🎯 Learning Style Optimization:**
{self._get_learning_style_tips(profile['learning_style'])}

**🧠 Knowledge Graph Strategy:**
• Focus on concepts with low mastery scores
• Build connections between related topics
• Use spaced repetition for retention

**🎮 Hobby Integration:**
• Connect new concepts to your interests in {', '.join(profile['hobbies'])}
• Create projects that combine learning with fun
• Use familiar contexts to understand abstract ideas

**📈 Progress Tracking:**
Your knowledge graph updates with every interaction, helping me provide better recommendations!"""

    def _generate_greeting_response(self, profile):
        """Generate personalized greeting"""
        return f"""👋 **Hello {profile['name']}!**

I'm your comprehensive AI learning assistant, powered by knowledge graph intelligence!

**What I can help you with:**
• 🧠 **Knowledge Graph Insights** - Track your learning progress
• 🔬 **Adaptive Labs** - Hands-on exercises tailored to you
• 📝 **Intelligent Quizzes** - Assessments that adapt to your level
• 🎨 **Hobby Integration** - Connect {', '.join(profile['hobbies'])} to {profile['current_focus']}
• 📚 **Study Guidance** - Personalized tips for your {profile['learning_style']} style

**Current Focus:** {profile['current_focus']}
**Learning Style:** {profile['learning_style']}

How can I help accelerate your learning journey today?"""

    def _generate_educational_response(self, user_message, profile):
        """Generate general educational response"""
        return f"""🎓 **Educational Assistance for {profile['name']}**

I'm here to help with your question about: "{user_message}"

**My Approach:**
• Tailored to your **{profile['learning_style']}** learning style
• Connected to your interests in {', '.join(profile['hobbies'])}
• Aligned with your focus on **{profile['current_focus']}**
• Informed by your knowledge graph progress

**What I can provide:**
• Detailed explanations with examples
• Practical applications and exercises
• Connections to your existing knowledge
• Next steps for deeper learning

Would you like me to break this down into smaller, manageable concepts or provide specific examples related to your interests?"""

    def _get_hobby_lab_examples(self, profile):
        """Get hobby-specific lab examples"""
        hobby_labs = []
        for hobby in profile['hobbies']:
            if hobby.lower() == 'gaming':
                hobby_labs.append("• 🎮 **Gaming AI Lab** - Build game recommendation systems")
            elif hobby.lower() == 'photography':
                hobby_labs.append("• 📸 **Photography ML Lab** - Image classification and enhancement")
            elif hobby.lower() == 'music':
                hobby_labs.append("• 🎵 **Music Analysis Lab** - Audio processing and recommendation")
            elif hobby.lower() == 'soccer':
                hobby_labs.append("• ⚽ **Sports Analytics Lab** - Player performance prediction")
            elif hobby.lower() == 'cooking':
                hobby_labs.append("• 👨‍🍳 **Recipe Optimization Lab** - Nutritional analysis and recommendations")

        return '\n'.join(hobby_labs) if hobby_labs else "• Custom labs based on your unique interests"

    def _get_hobby_connections(self, profile):
        """Get hobby connections to learning"""
        connections = []
        focus = profile['current_focus'].lower()

        for hobby in profile['hobbies']:
            if hobby.lower() == 'gaming' and 'machine learning' in focus:
                connections.append("• 🎮 **Gaming** → AI game agents, recommendation systems, player behavior analysis")
            elif hobby.lower() == 'photography' and any(term in focus for term in ['machine learning', 'ai', 'data']):
                connections.append("• 📸 **Photography** → Computer vision, image processing, neural networks")
            elif hobby.lower() == 'music':
                connections.append("• 🎵 **Music** → Audio signal processing, pattern recognition, recommendation algorithms")
            elif hobby.lower() == 'soccer' and 'data' in focus:
                connections.append("• ⚽ **Soccer** → Sports analytics, performance metrics, predictive modeling")
            elif hobby.lower() == 'cooking':
                connections.append("• 👨‍🍳 **Cooking** → Optimization algorithms, nutritional data analysis, recipe recommendations")

        return '\n'.join(connections) if connections else "• Your hobbies can be connected to learning in creative ways!"

    def _get_learning_style_tips(self, learning_style):
        """Get learning style specific tips"""
        style_lower = learning_style.lower()

        if 'visual' in style_lower:
            return """• Use diagrams, charts, and visual representations
• Create mind maps for concept connections
• Watch video tutorials and demonstrations
• Use color coding for different topics"""
        elif 'hands-on' in style_lower or 'kinesthetic' in style_lower:
            return """• Focus on practical labs and exercises
• Build projects to apply concepts
• Use interactive simulations
• Take frequent breaks for physical activity"""
        elif 'auditory' in style_lower:
            return """• Listen to educational podcasts
• Discuss concepts with others
• Use text-to-speech for reading
• Record yourself explaining concepts"""
        else:
            return """• Combine multiple learning approaches
• Adapt techniques based on the topic
• Experiment with different methods
• Focus on what works best for you"""

class KnowledgeGraphLabGenerator:
    """Generate accessible labs based on knowledge graph structure"""

    def __init__(self, foundational_kg, knowledge_components):
        self.kg = foundational_kg
        self.kcs = knowledge_components or {}

    def generate_accessible_labs(self, student_profile, student_kg=None):
        """Generate labs based on student's knowledge graph state"""
        labs = []

        # Get student's current mastery levels from knowledge graph
        mastery_levels = self._get_student_mastery(student_kg) if student_kg else {}

        # Generate labs for different difficulty levels
        beginner_labs = self._generate_beginner_labs(student_profile, mastery_levels)
        intermediate_labs = self._generate_intermediate_labs(student_profile, mastery_levels)
        advanced_labs = self._generate_advanced_labs(student_profile, mastery_levels)

        # Combine and prioritize based on student level
        level = student_profile.get('level', 'Beginner').lower()
        if 'beginner' in level:
            labs.extend(beginner_labs[:3])
            labs.extend(intermediate_labs[:1])
        elif 'intermediate' in level:
            labs.extend(intermediate_labs[:2])
            labs.extend(beginner_labs[:1])
            labs.extend(advanced_labs[:1])
        elif 'advanced' in level or 'expert' in level:
            labs.extend(advanced_labs[:2])
            labs.extend(intermediate_labs[:1])

        return labs[:4]  # Return top 4 labs

    def _get_student_mastery(self, student_kg):
        """Extract mastery levels from student knowledge graph"""
        mastery = {}
        if student_kg and hasattr(student_kg, 'graph'):
            for node in student_kg.graph.nodes():
                if student_kg.graph.nodes[node].get('type') == 'kc':
                    # Get mastery from edge to student
                    if student_kg.graph.has_edge(student_kg.student_id, node):
                        mastery[node] = student_kg.graph.edges[student_kg.student_id, node].get('mastery', 0.1)
        return mastery

    def _generate_beginner_labs(self, profile, mastery):
        """Generate beginner-level labs"""
        return [
            {
                'title': f'Introduction to {profile["specialty"]}',
                'description': f'Basic concepts and fundamentals of {profile["specialty"]}',
                'difficulty': 'Beginner',
                'time': '2-3 hours',
                'xp_reward': 100,
                'knowledge_components': ['basics', 'fundamentals'],
                'accessibility': 'High - Step-by-step guidance',
                'hobby_connection': self._get_hobby_connection(profile.get('hobbies', []), 'beginner')
            },
            {
                'title': 'Hands-on Practice Lab',
                'description': 'Interactive exercises to reinforce basic concepts',
                'difficulty': 'Beginner',
                'time': '1-2 hours',
                'xp_reward': 75,
                'knowledge_components': ['practice', 'application'],
                'accessibility': 'High - Interactive tutorials',
                'hobby_connection': self._get_hobby_connection(profile.get('hobbies', []), 'practice')
            }
        ]

    def _generate_intermediate_labs(self, profile, mastery):
        """Generate intermediate-level labs"""
        return [
            {
                'title': f'Advanced {profile["current_focus"]} Project',
                'description': f'Build a comprehensive project using {profile["current_focus"]} concepts',
                'difficulty': 'Intermediate',
                'time': '4-6 hours',
                'xp_reward': 200,
                'knowledge_components': ['project_based', 'integration'],
                'accessibility': 'Medium - Guided project structure',
                'hobby_connection': self._get_hobby_connection(profile.get('hobbies', []), 'project')
            },
            {
                'title': 'Real-world Application Lab',
                'description': 'Apply concepts to solve real-world problems',
                'difficulty': 'Intermediate',
                'time': '3-5 hours',
                'xp_reward': 150,
                'knowledge_components': ['application', 'problem_solving'],
                'accessibility': 'Medium - Problem-solving framework',
                'hobby_connection': self._get_hobby_connection(profile.get('hobbies', []), 'application')
            }
        ]

    def _generate_advanced_labs(self, profile, mastery):
        """Generate advanced-level labs"""
        return [
            {
                'title': f'Research-Level {profile["specialty"]} Challenge',
                'description': 'Cutting-edge research problems and methodologies',
                'difficulty': 'Advanced',
                'time': '8-12 hours',
                'xp_reward': 400,
                'knowledge_components': ['research', 'innovation'],
                'accessibility': 'Low - Independent research required',
                'hobby_connection': self._get_hobby_connection(profile.get('hobbies', []), 'research')
            },
            {
                'title': 'Industry Collaboration Project',
                'description': 'Work on real industry problems with professional standards',
                'difficulty': 'Expert',
                'time': '10-15 hours',
                'xp_reward': 500,
                'knowledge_components': ['professional', 'industry'],
                'accessibility': 'Low - Professional-level expectations',
                'hobby_connection': self._get_hobby_connection(profile.get('hobbies', []), 'professional')
            }
        ]

    def _get_hobby_connection(self, hobbies, context):
        """Connect labs to student hobbies"""
        connections = {
            '🎮 Gaming': {
                'beginner': 'Learn programming through game development basics',
                'practice': 'Create simple games to practice coding',
                'project': 'Build a complete game with AI features',
                'application': 'Develop game analytics and player behavior systems',
                'research': 'Research advanced game AI and procedural generation',
                'professional': 'Industry-standard game development practices'
            },
            '📸 Photography': {
                'beginner': 'Learn image processing fundamentals',
                'practice': 'Apply filters and effects to photos',
                'project': 'Build a photo enhancement application',
                'application': 'Create computer vision systems for photography',
                'research': 'Research advanced image recognition techniques',
                'professional': 'Professional photo editing software development'
            },
            '⚽ Soccer': {
                'beginner': 'Learn data analysis with sports statistics',
                'practice': 'Analyze player performance data',
                'project': 'Build a sports analytics dashboard',
                'application': 'Create predictive models for game outcomes',
                'research': 'Research advanced sports analytics techniques',
                'professional': 'Professional sports data science applications'
            }
        }

        for hobby in hobbies:
            if hobby in connections and context in connections[hobby]:
                return connections[hobby][context]

        return f'Apply concepts to your interests in {", ".join(hobbies[:2])}'

class KnowledgeGraphQuizGenerator:
    """Generate accessible quizzes based on knowledge graph structure"""

    def __init__(self, foundational_kg, knowledge_components):
        self.kg = foundational_kg
        self.kcs = knowledge_components or {}

    def generate_accessible_quizzes(self, student_profile, student_kg=None):
        """Generate quizzes based on student's knowledge graph state"""
        quizzes = []

        # Get student's current mastery levels
        mastery_levels = self._get_student_mastery(student_kg) if student_kg else {}

        # Generate different types of quizzes
        diagnostic_quiz = self._generate_diagnostic_quiz(student_profile, mastery_levels)
        practice_quiz = self._generate_practice_quiz(student_profile, mastery_levels)
        challenge_quiz = self._generate_challenge_quiz(student_profile, mastery_levels)
        hobby_quiz = self._generate_hobby_based_quiz(student_profile, mastery_levels)

        quizzes.extend([diagnostic_quiz, practice_quiz, challenge_quiz, hobby_quiz])
        return quizzes

    def _get_student_mastery(self, student_kg):
        """Extract mastery levels from student knowledge graph"""
        mastery = {}
        if student_kg and hasattr(student_kg, 'graph'):
            for node in student_kg.graph.nodes():
                if student_kg.graph.nodes[node].get('type') == 'kc':
                    if student_kg.graph.has_edge(student_kg.student_id, node):
                        mastery[node] = student_kg.graph.edges[student_kg.student_id, node].get('mastery', 0.1)
        return mastery

    def _generate_diagnostic_quiz(self, profile, mastery):
        """Generate diagnostic quiz to assess current knowledge"""
        return {
            'title': f'{profile["specialty"]} Knowledge Assessment',
            'description': 'Diagnostic quiz to identify your current knowledge level and learning gaps',
            'type': 'diagnostic',
            'questions': 15,
            'time': '20-25 minutes',
            'difficulty': 'Adaptive',
            'xp_reward': 50,
            'accessibility': 'High - Adaptive difficulty based on responses',
            'knowledge_components': list(mastery.keys())[:5] if mastery else ['fundamentals'],
            'hobby_connection': self._get_hobby_quiz_connection(profile.get('hobbies', []), 'assessment')
        }

    def _generate_practice_quiz(self, profile, mastery):
        """Generate practice quiz for reinforcement"""
        return {
            'title': f'{profile["current_focus"]} Practice Quiz',
            'description': 'Reinforcement quiz to practice and strengthen your understanding',
            'type': 'practice',
            'questions': 10,
            'time': '15-20 minutes',
            'difficulty': profile.get('level', 'Beginner'),
            'xp_reward': 75,
            'accessibility': 'High - Immediate feedback and explanations',
            'knowledge_components': [profile.get('current_focus', 'general')],
            'hobby_connection': self._get_hobby_quiz_connection(profile.get('hobbies', []), 'practice')
        }

    def _generate_challenge_quiz(self, profile, mastery):
        """Generate challenging quiz for advanced learners"""
        return {
            'title': f'Advanced {profile["specialty"]} Challenge',
            'description': 'Challenging quiz to test your mastery and push your limits',
            'type': 'challenge',
            'questions': 12,
            'time': '25-30 minutes',
            'difficulty': 'Advanced',
            'xp_reward': 150,
            'accessibility': 'Medium - Complex scenarios and applications',
            'knowledge_components': ['advanced_concepts', 'application'],
            'hobby_connection': self._get_hobby_quiz_connection(profile.get('hobbies', []), 'challenge')
        }

    def _generate_hobby_based_quiz(self, profile, mastery):
        """Generate quiz connecting learning to hobbies"""
        hobbies = profile.get('hobbies', [])
        main_hobby = hobbies[0] if hobbies else 'General Interest'

        return {
            'title': f'{main_hobby} & {profile["specialty"]} Connection Quiz',
            'description': f'Explore how {profile["specialty"]} applies to your interest in {main_hobby}',
            'type': 'hobby_based',
            'questions': 8,
            'time': '12-15 minutes',
            'difficulty': 'Intermediate',
            'xp_reward': 100,
            'accessibility': 'High - Real-world applications and examples',
            'knowledge_components': ['application', 'real_world'],
            'hobby_connection': f'Direct connection between {main_hobby} and {profile["specialty"]}'
        }

    def _get_hobby_quiz_connection(self, hobbies, quiz_type):
        """Connect quizzes to student hobbies"""
        connections = {
            '🎮 Gaming': {
                'assessment': 'Questions about game development and AI in gaming',
                'practice': 'Practice problems using game scenarios',
                'challenge': 'Complex game AI and development challenges'
            },
            '📸 Photography': {
                'assessment': 'Questions about image processing and computer vision',
                'practice': 'Practice with photo editing and enhancement problems',
                'challenge': 'Advanced computer vision and image analysis challenges'
            },
            '⚽ Soccer': {
                'assessment': 'Questions about sports analytics and data science',
                'practice': 'Practice with sports data analysis problems',
                'challenge': 'Complex sports prediction and analytics challenges'
            }
        }

        for hobby in hobbies:
            if hobby in connections and quiz_type in connections[hobby]:
                return connections[hobby][quiz_type]

        return f'Questions related to your interests in {", ".join(hobbies[:2])}'

    def _build_comprehensive_knowledge_base(self):
        """Build comprehensive knowledge base integrating knowledge graph data"""
        base_kb = {
            'machine_learning': {
                'concepts': ['neural networks', 'deep learning', 'supervised learning', 'unsupervised learning', 'reinforcement learning'],
                'applications': ['computer vision', 'natural language processing', 'recommendation systems', 'autonomous vehicles'],
                'tools': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy'],
                'knowledge_graph_concepts': []
            },
            'data_science': {
                'concepts': ['statistics', 'data visualization', 'feature engineering', 'model evaluation', 'cross-validation'],
                'applications': ['business analytics', 'predictive modeling', 'data mining', 'statistical analysis'],
                'tools': ['python', 'r', 'sql', 'tableau', 'power bi', 'jupyter'],
                'knowledge_graph_concepts': []
            },
            'programming': {
                'concepts': ['variables', 'functions', 'loops', 'conditionals', 'classes', 'objects'],
                'languages': ['python', 'javascript', 'java', 'c++', 'r', 'sql'],
                'paradigms': ['object-oriented', 'functional', 'procedural'],
                'knowledge_graph_concepts': []
            },
            'ai_research': {
                'concepts': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
                'applications': ['computer vision', 'nlp', 'robotics', 'autonomous systems'],
                'tools': ['tensorflow', 'pytorch', 'opencv', 'nltk'],
                'knowledge_graph_concepts': []
            },
            'software_engineering': {
                'concepts': ['software design', 'algorithms', 'data structures', 'system architecture'],
                'practices': ['agile', 'devops', 'testing', 'version control'],
                'tools': ['git', 'docker', 'kubernetes', 'jenkins'],
                'knowledge_graph_concepts': []
            }
        }

        # Integrate knowledge graph concepts if available
        if self.knowledge_components:
            for category in base_kb:
                base_kb[category]['knowledge_graph_concepts'] = list(self.knowledge_components.keys())[:10]

        return base_kb

    def generate_comprehensive_response(self, message, student_id, students_data):
        """Generate intelligent, knowledge graph-aware responses"""
        context = self.get_student_context(student_id, students_data)
        if not context:
            return "I'm sorry, I couldn't find your student profile. Please try again."

        # Get student's knowledge graph if available
        student_kg = None
        if student_id in students_data and 'knowledge_graph' in students_data[student_id]:
            student_kg = students_data[student_id]['knowledge_graph']

        # Initialize conversation history for new students
        if student_id not in self.conversation_history:
            self.conversation_history[student_id] = []

        # Add user message to history
        self.conversation_history[student_id].append({
            'type': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        # Analyze message and generate response
        response = self._generate_contextual_response(message, context, student_kg)

        # Add bot response to history
        self.conversation_history[student_id].append({
            'type': 'bot',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })

        return response

    def _generate_contextual_response(self, message, context, student_kg):
        """Generate contextual response using knowledge graph data"""
        message_lower = message.lower()
        name = context['name']

        # Knowledge graph-aware responses
        if student_kg and any(word in message_lower for word in ['progress', 'mastery', 'knowledge', 'learning']):
            return self._generate_knowledge_graph_response(message, context, student_kg)

        # Lab and quiz generation requests
        if any(word in message_lower for word in ['lab', 'project', 'hands-on', 'practice']):
            return self._generate_lab_recommendations(context, student_kg)

        if any(word in message_lower for word in ['quiz', 'test', 'assessment', 'check']):
            return self._generate_quiz_recommendations(context, student_kg)

        # Greeting responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'start']):
            return f"Hello {name}! 👋 I'm your comprehensive AI learning assistant with access to your knowledge graph. I can help you with:\n\n🧠 **Knowledge Assessment** - Track your learning progress\n📚 **Personalized Labs** - Hands-on projects based on your interests\n📝 **Adaptive Quizzes** - Assessments that match your level\n🎯 **Study Guidance** - Tailored to your {context['specialty']} focus\n\nWhat would you like to explore today?"

        # Study tips and learning strategies
        elif any(word in message_lower for word in ['study', 'learn', 'tips', 'strategy', 'how to']):
            return self._generate_study_tips_response(context, student_kg)

        # Hobby connections
        elif any(word in message_lower for word in ['hobby', 'interest', 'gaming', 'photography', 'music', 'soccer']):
            return self._generate_hobby_connection_response(message, context)

        # Help and support
        elif any(word in message_lower for word in ['help', 'stuck', 'confused', 'difficult']):
            return self._generate_help_response(context, student_kg)

        # Default comprehensive response
        else:
            return self._generate_general_response(message, context, student_kg)

    def _generate_knowledge_graph_response(self, message, context, student_kg):
        """Generate response based on knowledge graph data"""
        if not student_kg:
            return f"I'd love to help you track your progress, {context['name']}! While I don't have access to your detailed knowledge graph right now, I can see you're working on {context['current_focus']}. Would you like me to suggest some practice activities or assessments to help gauge your understanding?"

        # Extract mastery information from knowledge graph
        mastery_info = self._extract_mastery_info(student_kg)

        return f"Great question about your learning progress, {context['name']}! 📊\n\nBased on your knowledge graph:\n\n🎯 **Current Focus**: {context['current_focus']}\n📈 **Overall Progress**: {mastery_info['overall']}%\n🔥 **Strongest Areas**: {', '.join(mastery_info['strong_areas'])}\n💪 **Growth Opportunities**: {', '.join(mastery_info['growth_areas'])}\n\nWould you like me to suggest specific labs or quizzes to strengthen your weaker areas?"

    def _extract_mastery_info(self, student_kg):
        """Extract mastery information from student knowledge graph"""
        if not student_kg or not hasattr(student_kg, 'graph'):
            return {
                'overall': 65,
                'strong_areas': ['Fundamentals', 'Basic Concepts'],
                'growth_areas': ['Advanced Topics', 'Applications']
            }

        mastery_levels = {}
        for node in student_kg.graph.nodes():
            if student_kg.graph.nodes[node].get('type') == 'kc':
                if student_kg.graph.has_edge(student_kg.student_id, node):
                    mastery = student_kg.graph.edges[student_kg.student_id, node].get('mastery', 0.1)
                    name = student_kg.graph.nodes[node].get('name', node)
                    mastery_levels[name] = mastery

        if not mastery_levels:
            return {
                'overall': 65,
                'strong_areas': ['Fundamentals'],
                'growth_areas': ['Advanced Topics']
            }

        overall = int(sum(mastery_levels.values()) / len(mastery_levels) * 100)
        sorted_mastery = sorted(mastery_levels.items(), key=lambda x: x[1], reverse=True)

        return {
            'overall': overall,
            'strong_areas': [item[0] for item in sorted_mastery[:2]],
            'growth_areas': [item[0] for item in sorted_mastery[-2:]]
        }

    def _generate_lab_recommendations(self, context, student_kg):
        """Generate lab recommendations using knowledge graph"""
        labs = self.lab_generator.generate_accessible_labs(context, student_kg)

        response = f"Perfect! Here are some personalized labs for you, {context['name']}! 🔬\n\n"

        for i, lab in enumerate(labs[:3], 1):
            response += f"**{i}. {lab['title']}**\n"
            response += f"   📝 {lab['description']}\n"
            response += f"   ⏱️ Time: {lab['time']}\n"
            response += f"   🎯 Difficulty: {lab['difficulty']}\n"
            response += f"   ♿ Accessibility: {lab['accessibility']}\n"
            response += f"   🎮 Connection: {lab['hobby_connection']}\n\n"

        response += "Which lab interests you most? I can provide more details or suggest similar alternatives!"
        return response

    def _generate_quiz_recommendations(self, context, student_kg):
        """Generate quiz recommendations using knowledge graph"""
        quizzes = self.quiz_generator.generate_accessible_quizzes(context, student_kg)

        response = f"Great idea! Here are some personalized quizzes for you, {context['name']}! 📝\n\n"

        for i, quiz in enumerate(quizzes[:3], 1):
            response += f"**{i}. {quiz['title']}**\n"
            response += f"   📋 {quiz['description']}\n"
            response += f"   ❓ Questions: {quiz['questions']}\n"
            response += f"   ⏱️ Time: {quiz['time']}\n"
            response += f"   🎯 Difficulty: {quiz['difficulty']}\n"
            response += f"   ♿ Accessibility: {quiz['accessibility']}\n\n"

        response += "Which quiz would you like to try? I can explain more about any of these assessments!"
        return response

    def _generate_study_tips_response(self, context, student_kg):
        """Generate personalized study tips"""
        learning_style = context.get('learning_style', 'Visual').lower()
        name = context['name']

        if 'visual' in learning_style:
            tips = [
                "📊 Create mind maps and concept diagrams",
                "🎨 Use color-coding for different topics",
                "📈 Draw flowcharts for complex processes",
                "🖼️ Watch educational videos and tutorials",
                "📱 Use visual learning apps and tools"
            ]
        elif 'auditory' in learning_style:
            tips = [
                "🎧 Listen to educational podcasts",
                "🗣️ Explain concepts out loud to yourself",
                "👥 Join study groups and discussions",
                "🎵 Use mnemonics and memory songs",
                "📻 Record yourself explaining topics"
            ]
        elif 'kinesthetic' in learning_style:
            tips = [
                "🤲 Use hands-on projects and experiments",
                "🚶 Take walking breaks while studying",
                "🎯 Practice with real-world applications",
                "🔧 Build physical models or prototypes",
                "⚡ Use movement-based memory techniques"
            ]
        else:
            tips = [
                "📚 Combine multiple learning methods",
                "🎯 Set specific, achievable goals",
                "⏰ Use the Pomodoro Technique",
                "🔄 Review material regularly",
                "🤝 Teach concepts to others"
            ]

        response = f"Excellent question, {name}! As a {learning_style} learner studying {context['current_focus']}, here are personalized tips:\n\n"
        response += "\n".join(tips[:4])
        response += f"\n\n💡 **Bonus tip**: Connect {context['current_focus']} to your hobbies like {', '.join(context.get('hobbies', [])[:2])} for better retention!"

        return response

    def _generate_hobby_connection_response(self, message, context):
        """Generate hobby connection responses"""
        hobbies = context.get('hobbies', [])
        specialty = context['specialty']
        name = context['name']

        connections = {
            '🎮 Gaming': f"Gaming and {specialty} have amazing synergies! You can apply {specialty} to game AI, procedural generation, player analytics, and creating intelligent NPCs.",
            '📸 Photography': f"Photography connects beautifully with {specialty}! Think computer vision, image processing, automatic enhancement, object recognition, and AI-powered editing tools.",
            '🎵 Music Production': f"Music and {specialty} create incredible possibilities! AI music generation, audio processing, recommendation systems, and sound analysis are just the beginning.",
            '⚽ Soccer': f"Sports analytics is a perfect application of {specialty}! Player performance analysis, game prediction, tactical analysis, and injury prevention modeling.",
            '🍳 Cooking': f"Culinary arts meets {specialty} in fascinating ways! Recipe optimization, nutrition analysis, flavor pairing algorithms, and smart kitchen systems.",
            '📚 Reading': f"Literature and {specialty} intersect in text analysis, sentiment analysis, recommendation systems, and natural language processing applications."
        }

        response = f"Great question, {name}! 🌟\n\n"

        for hobby in hobbies[:2]:
            if hobby in connections:
                response += f"**{hobby} & {specialty}:**\n{connections[hobby]}\n\n"

        response += f"Would you like me to suggest specific projects that combine your interests with {specialty}?"

        return response

    def _generate_help_response(self, context, student_kg):
        """Generate helpful support responses"""
        name = context['name']
        current_focus = context['current_focus']

        return f"I'm here to help, {name}! 💪 Learning {current_focus} can be challenging, but you've got this!\n\n🌟 **Remember**: Every expert was once a beginner\n📈 **Focus on**: Small, consistent improvements\n🎯 **Strategy**: Break complex problems into smaller steps\n\n**I can help you with:**\n• Explaining difficult concepts step-by-step\n• Suggesting practice exercises\n• Connecting topics to your interests\n• Finding additional resources\n• Creating a personalized study plan\n\nWhat specific part is giving you trouble? I'm here to guide you through it!"

    def _generate_general_response(self, message, context, student_kg):
        """Generate general contextual responses"""
        name = context['name']
        specialty = context['specialty']

        return f"Thanks for your question, {name}! I'm here to help you with {specialty} and connect your learning to your interests.\n\n🤖 **I can assist with:**\n• Personalized explanations and tutorials\n• Practice labs and projects\n• Knowledge assessments and quizzes\n• Study strategies and tips\n• Connecting concepts to your hobbies\n\nCould you tell me more about what you'd like to explore or any specific challenges you're facing?"

    def get_student_context(self, student_id, students_data):
        """Get comprehensive student context"""
        if student_id not in students_data:
            return None

        student_data = students_data[student_id]
        profile = student_data['profile']

        return {
            'name': profile['name'],
            'specialty': profile['specialty'],
            'level': profile['level'],
            'current_focus': profile['current_focus'],
            'learning_style': profile['learning_style'],
            'hobbies': profile.get('hobbies', []),
            'streak_days': profile.get('streak_days', 0),
            'knowledge_graph': student_data.get('knowledge_graph')
        }

    def _build_knowledge_base(self):
        """Build comprehensive knowledge base for educational topics"""
        return {
            'machine_learning': {
                'concepts': ['neural networks', 'deep learning', 'supervised learning', 'unsupervised learning', 'reinforcement learning'],
                'applications': ['computer vision', 'natural language processing', 'recommendation systems', 'autonomous vehicles'],
                'tools': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy']
            },
            'data_science': {
                'concepts': ['statistics', 'data visualization', 'feature engineering', 'model evaluation', 'cross-validation'],
                'applications': ['business analytics', 'predictive modeling', 'data mining', 'statistical analysis'],
                'tools': ['python', 'r', 'sql', 'tableau', 'power bi', 'jupyter']
            },
            'programming': {
                'concepts': ['algorithms', 'data structures', 'object-oriented programming', 'functional programming', 'design patterns'],
                'applications': ['web development', 'mobile apps', 'system programming', 'game development'],
                'tools': ['python', 'javascript', 'java', 'c++', 'react', 'node.js']
            },
            'ai_research': {
                'concepts': ['artificial intelligence', 'cognitive science', 'neural networks', 'expert systems', 'knowledge representation'],
                'applications': ['robotics', 'natural language understanding', 'computer vision', 'automated reasoning'],
                'tools': ['tensorflow', 'pytorch', 'opencv', 'nltk', 'spacy', 'transformers']
            },
            'software_engineering': {
                'concepts': ['software architecture', 'design patterns', 'testing', 'version control', 'agile methodology'],
                'applications': ['web applications', 'mobile development', 'cloud computing', 'microservices'],
                'tools': ['git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure']
            }
        }

    def get_student_context(self, student_id, students_data):
        """Get student context for personalized responses"""
        if student_id not in students_data:
            return None

        profile = students_data[student_id]['profile']
        return {
            'name': profile['name'],
            'level': profile['level'],
            'specialty': profile['specialty'],
            'hobbies': profile['hobbies'],
            'favorite_topics': profile['favorite_topics'],
            'current_focus': profile['current_focus'],
            'learning_style': profile['learning_style']
        }

    def generate_response(self, message, student_id, students_data):
        """Generate intelligent, personalized responses"""
        context = self.get_student_context(student_id, students_data)
        if not context:
            return "I'm sorry, I couldn't find your student profile. Please try again."

        # Initialize conversation history for new students
        if student_id not in self.conversation_history:
            self.conversation_history[student_id] = []

        # Add user message to history
        self.conversation_history[student_id].append({
            'type': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        # Generate response based on message content and student context
        response = self._process_message(message.lower(), context)

        # Add bot response to history
        self.conversation_history[student_id].append({
            'type': 'bot',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })

        return response

    def _process_message(self, message, context):
        """Process message and generate appropriate response"""
        name = context['name'].split()[0]  # First name
        specialty = context['specialty'].lower()
        level = context['level'].lower()
        hobbies = [hobby.lower() for hobby in context['hobbies']]

        # Greeting responses
        if any(word in message for word in ['hello', 'hi', 'hey', 'greetings']):
            return f"Hello {name}! 👋 I'm your AI learning assistant. I'm here to help you with {context['specialty']} and answer any questions you have. What would you like to explore today?"

        # Help requests
        elif any(word in message for word in ['help', 'assist', 'support']):
            return f"I'm here to help you succeed in {context['specialty']}! I can:\n\n🎯 Answer questions about your current focus: {context['current_focus']}\n📚 Explain concepts in {specialty}\n🔍 Suggest learning resources\n🎮 Connect learning to your hobbies\n💡 Provide study tips for {context['learning_style']} learners\n\nWhat specific topic would you like help with?"

        # Study tips and learning strategies
        elif any(word in message for word in ['study', 'learn', 'tips', 'strategy', 'how to']):
            learning_style = context['learning_style'].lower()
            if 'visual' in learning_style:
                return f"Great question, {name}! As a visual learner, here are some effective strategies:\n\n📊 Create mind maps and diagrams\n🎨 Use color-coding for different concepts\n📈 Draw flowcharts for processes\n🖼️ Watch educational videos and tutorials\n📱 Use visual learning apps\n\nFor {context['current_focus']}, try creating visual representations of the key concepts. Would you like specific visualization techniques for this topic?"
            elif 'auditory' in learning_style:
                return f"Perfect, {name}! As an auditory learner, try these techniques:\n\n🎧 Listen to educational podcasts\n🗣️ Explain concepts out loud\n👥 Join study groups for discussions\n🎵 Create mnemonics or songs\n📻 Record yourself explaining topics\n\nFor {context['current_focus']}, consider finding audio resources or discussing with peers. Want me to suggest some specific audio learning resources?"
            else:
                return f"Here are some effective learning strategies for {context['current_focus']}, {name}:\n\n📝 Active note-taking\n🔄 Spaced repetition\n🎯 Practice with real examples\n👥 Teach others to reinforce learning\n🧩 Break complex topics into smaller parts\n\nWhat specific aspect would you like to focus on?"

        # Hobby-related learning connections
        elif any(hobby in message for hobby in hobbies):
            matching_hobby = next(hobby for hobby in hobbies if hobby in message)
            return self._generate_hobby_connection(matching_hobby, context)

        # Concept explanations
        elif any(word in message for word in ['what is', 'explain', 'define', 'meaning']):
            return self._explain_concept(message, context)

        # Progress and motivation
        elif any(word in message for word in ['progress', 'motivation', 'stuck', 'difficult', 'hard']):
            return f"I understand, {name}! Learning {context['current_focus']} can be challenging, but you're making great progress! 💪\n\n🌟 Remember: Every expert was once a beginner\n📈 Focus on small, consistent improvements\n🎯 Break complex problems into smaller steps\n🏆 Celebrate your achievements along the way\n\nAs a {level} learner, you have the foundation to tackle this. What specific part is giving you trouble? I can help break it down!"

        # Resource requests
        elif any(word in message for word in ['resource', 'book', 'tutorial', 'course', 'material']):
            return self._suggest_resources(context)

        # Project ideas
        elif any(word in message for word in ['project', 'practice', 'build', 'create']):
            return self._suggest_projects(context)

        # Default intelligent response
        else:
            return f"That's an interesting question, {name}! Based on your focus on {context['current_focus']} and your {level} level, let me help you with that.\n\n🤔 Could you provide a bit more detail about what specifically you'd like to know?\n\nI can help with:\n• Concept explanations\n• Study strategies\n• Project ideas\n• Resource recommendations\n• Connecting learning to your hobbies\n\nWhat would be most helpful for you right now?"

    def _generate_hobby_connection(self, hobby, context):
        """Generate responses connecting hobbies to learning"""
        name = context['name'].split()[0]
        specialty = context['specialty']

        hobby_connections = {
            'gaming': f"Awesome, {name}! Gaming and {specialty} have amazing connections:\n\n🎮 Game AI development\n🧠 Procedural content generation\n📊 Player behavior analytics\n🎯 Recommendation systems for games\n⚡ Real-time decision making\n\nYour gaming experience gives you intuition for algorithms and user experience. Want to explore how to build AI for games?",
            'photography': f"Perfect, {name}! Photography and {specialty} are beautifully connected:\n\n📸 Computer vision and image processing\n🎨 Style transfer and filters\n🔍 Object detection and recognition\n📊 Image analytics and metadata\n🌟 Automated photo enhancement\n\nYour eye for composition translates well to data visualization! Interested in building photo AI tools?",
            'music': f"Fantastic, {name}! Music and {specialty} create beautiful harmony:\n\n🎵 Audio signal processing\n🤖 Music generation algorithms\n📊 Music recommendation systems\n🎼 Pattern recognition in compositions\n🔊 Sound analysis and synthesis\n\nYour musical background helps with pattern recognition and creativity in problem-solving! Want to explore music AI?",
            'soccer': f"Great connection, {name}! Soccer and {specialty} make a winning team:\n\n⚽ Player performance analytics\n📊 Game strategy optimization\n🎯 Predictive modeling for outcomes\n📈 Injury prevention through data\n🏆 Team formation algorithms\n\nYour understanding of strategy and teamwork applies perfectly to data science! Interested in sports analytics?",
            'chess': f"Excellent, {name}! Chess and {specialty} are strategic partners:\n\n♟️ Game tree algorithms\n🧠 Minimax and alpha-beta pruning\n🤖 AI decision making\n📊 Position evaluation functions\n🎯 Pattern recognition\n\nYour strategic thinking from chess is perfect for algorithm design! Want to build a chess AI?",
            'space': f"Amazing, {name}! Space exploration and {specialty} reach for the stars:\n\n🚀 Orbital mechanics simulation\n🛰️ Satellite data processing\n🌌 Astronomical image analysis\n📡 Mission planning algorithms\n🔭 Data from space telescopes\n\nYour curiosity about space drives innovation in computational modeling! Interested in space data science?"
        }

        # Find matching hobby connection
        for key, response in hobby_connections.items():
            if key in hobby.lower():
                return response

        return f"I love that you're into {hobby}, {name}! There are definitely ways to connect this interest with {specialty}. Let's explore how your passion can enhance your learning experience!"

    def _explain_concept(self, message, context):
        """Explain concepts based on student's level and specialty"""
        name = context['name'].split()[0]
        level = context['level'].lower()
        specialty = context['specialty'].lower()

        # Extract potential concept from message
        concept_keywords = {
            'machine learning': "Machine Learning is like teaching computers to learn patterns from data, just like how you learn to recognize faces or predict outcomes. It's the foundation of AI that powers recommendations, image recognition, and much more!",
            'neural network': "Neural Networks are inspired by how our brain works! They're networks of connected nodes (like neurons) that process information. Each connection has a weight that gets adjusted as the network learns from examples.",
            'algorithm': "An Algorithm is like a recipe for solving problems! It's a step-by-step set of instructions that tells a computer exactly what to do. Just like following a cooking recipe, algorithms help us solve complex problems systematically.",
            'data science': "Data Science is like being a detective with numbers! You collect clues (data), analyze patterns, and solve mysteries (business problems). It combines statistics, programming, and domain knowledge to extract insights.",
            'deep learning': "Deep Learning uses neural networks with many layers (that's why it's 'deep'!) to automatically learn complex patterns. It's what powers image recognition, language translation, and even creative AI like art generation!"
        }

        for keyword, explanation in concept_keywords.items():
            if keyword in message.lower():
                if level == 'beginner':
                    return f"Great question, {name}! Let me explain {keyword} in simple terms:\n\n{explanation}\n\n🎯 Think of it as a tool that helps solve problems in {specialty}. Would you like me to give you a specific example related to your interests?"
                elif level == 'intermediate':
                    return f"Excellent question, {name}! Here's a deeper look at {keyword}:\n\n{explanation}\n\n🔍 In {specialty}, this concept is crucial for {context['current_focus']}. Want to explore how it applies to your current projects?"
                else:  # advanced/expert
                    return f"Sophisticated question, {name}! {keyword} in the context of {specialty}:\n\n{explanation}\n\n🚀 Given your advanced level, you might be interested in the latest research developments or implementation challenges. What specific aspect would you like to dive deeper into?"

        return f"That's a thoughtful question, {name}! I'd love to help explain that concept. Could you be more specific about which aspect of {context['current_focus']} you'd like me to clarify? I can tailor my explanation to your {level} level."

    def _suggest_resources(self, context):
        """Suggest learning resources based on student profile"""
        name = context['name'].split()[0]
        specialty = context['specialty'].lower()
        level = context['level'].lower()

        resources = {
            'machine learning': {
                'beginner': ['Coursera ML Course by Andrew Ng', 'Python Machine Learning by Sebastian Raschka', 'Kaggle Learn courses'],
                'intermediate': ['Hands-On Machine Learning by Aurélien Géron', 'Fast.ai courses', 'Papers With Code'],
                'advanced': ['Deep Learning by Ian Goodfellow', 'Latest arXiv papers', 'Google AI Research']
            },
            'data science': {
                'beginner': ['Python for Data Analysis by Wes McKinney', 'DataCamp courses', 'Kaggle datasets'],
                'intermediate': ['The Elements of Statistical Learning', 'Towards Data Science blog', 'Jupyter notebooks'],
                'advanced': ['Advanced Analytics with Spark', 'Research papers', 'Industry case studies']
            }
        }

        if specialty in resources and level in resources[specialty]:
            resource_list = resources[specialty][level]
            return f"Perfect timing, {name}! Here are some excellent resources for {level} {specialty} learning:\n\n" + \
                   "\n".join([f"📚 {resource}" for resource in resource_list]) + \
                   f"\n\n🎯 These align well with your current focus on {context['current_focus']}. Would you like specific recommendations for any of these?"

        return f"Great question, {name}! For {specialty} at your {level} level, I recommend starting with foundational courses and gradually moving to specialized resources. What specific area would you like resources for?"

    def _suggest_projects(self, context):
        """Suggest project ideas based on student profile"""
        name = context['name'].split()[0]
        specialty = context['specialty'].lower()
        hobbies = context['hobbies']

        # Generate hobby-connected project ideas
        project_ideas = []

        for hobby in hobbies[:2]:  # Focus on first 2 hobbies
            if 'gaming' in hobby.lower():
                project_ideas.append("🎮 Build a game AI that learns player behavior")
            elif 'photography' in hobby.lower():
                project_ideas.append("📸 Create an AI photo enhancement tool")
            elif 'music' in hobby.lower():
                project_ideas.append("🎵 Develop a music recommendation system")
            elif 'soccer' in hobby.lower():
                project_ideas.append("⚽ Analyze player performance data")
            elif 'chess' in hobby.lower():
                project_ideas.append("♟️ Build a chess position evaluator")

        # Add general projects based on specialty
        if 'machine learning' in specialty:
            project_ideas.append("🤖 Create a chatbot for your favorite topic")
            project_ideas.append("📊 Build a predictive model for real-world data")
        elif 'data science' in specialty:
            project_ideas.append("📈 Analyze trends in your area of interest")
            project_ideas.append("🔍 Create an interactive data dashboard")

        if project_ideas:
            return f"Exciting, {name}! Here are some project ideas that combine {specialty} with your interests:\n\n" + \
                   "\n".join(project_ideas) + \
                   f"\n\n🚀 These projects will help you apply {context['current_focus']} in practical ways. Which one sounds most interesting to you?"

        return f"Great initiative, {name}! Building projects is the best way to learn {specialty}. What type of problem or application interests you most? I can suggest specific projects based on your preferences!"

# Initialize the comprehensive AI system
comprehensive_ai = ComprehensiveBITTutorAI(foundational_kg, qm, kcs)

def generate_advanced_analytics(student_id, kg_data, profile):
    """Generate comprehensive analytics from knowledge graph data"""
    if not kg_data:
        return create_demo_student_data(student_id, profile)['analytics']
    
    # Extract real analytics from knowledge graph
    analytics = {
        'performance_trend': generate_performance_trend(),
        'skill_mastery': generate_skill_mastery(),
        'learning_velocity': generate_learning_velocity(),
        'engagement_patterns': generate_engagement_patterns(),
        'ai_insights': generate_ai_insights(profile),
        'next_recommendations': generate_smart_recommendations(profile)
    }
    
    return analytics

# Load student data on startup
load_student_data()
print(f"🎓 Loaded {len(students_data)} student profiles")

@app.route('/')
def nexus_home():
    """BIT Tutor Home - Student Selection Hub"""
    return render_template('nexus_home.html', students=students_data)

@app.route('/student/<student_id>')
def student_nexus(student_id):
    """Individual Student BIT Tutor Dashboard"""
    if student_id not in students_data:
        return redirect(url_for('nexus_home'))
    
    student = students_data[student_id]
    session['current_student'] = student_id
    
    return render_template('nexus_dashboard.html', 
                         student=student, 
                         student_id=student_id)

@app.route('/api/student/<student_id>/analytics')
def get_student_analytics(student_id):
    """API endpoint for real-time analytics"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404
    
    return jsonify(students_data[student_id]['analytics'])

@app.route('/api/student/<student_id>/live_metrics')
def get_live_metrics(student_id):
    """API endpoint for live dashboard metrics"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    # Simulate real-time metrics
    metrics = {
        'current_session_time': random.randint(15, 120),
        'concepts_learned_today': random.randint(2, 8),
        'accuracy_rate': random.uniform(85, 98),
        'focus_score': random.uniform(75, 95),
        'ai_confidence': random.uniform(88, 99),
        'next_milestone_progress': random.uniform(60, 85)
    }

    return jsonify(metrics)

@app.route('/api/student/<student_id>/personalized_labs')
def get_personalized_labs(student_id):
    """API endpoint for knowledge graph-based learning labs"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    profile = students_data[student_id]['profile']
    student_kg = students_data[student_id].get('knowledge_graph')

    # Generate labs using knowledge graph
    labs = comprehensive_ai.lab_generator.generate_accessible_labs(profile, student_kg)

    return jsonify({
        'labs': labs,
        'student_name': profile['name'],
        'knowledge_graph_enabled': student_kg is not None
    })

@app.route('/api/student/<student_id>/personalized_quizzes')
def get_personalized_quizzes(student_id):
    """API endpoint for knowledge graph-based quizzes"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    profile = students_data[student_id]['profile']
    student_kg = students_data[student_id].get('knowledge_graph')

    # Generate quizzes using knowledge graph
    quizzes = comprehensive_ai.quiz_generator.generate_accessible_quizzes(profile, student_kg)

    return jsonify({
        'quizzes': quizzes,
        'student_name': profile['name'],
        'knowledge_graph_enabled': student_kg is not None
    })

@app.route('/api/student/<student_id>/learning_materials')
def get_learning_materials(student_id):
    """API endpoint for knowledge graph-based learning materials"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    profile = students_data[student_id]['profile']
    student_kg = students_data[student_id].get('knowledge_graph')

    # Generate learning materials using knowledge graph
    materials = comprehensive_ai.materials_generator.generate_learning_materials(profile, student_kg)

    return jsonify({
        'materials': materials,
        'student_name': profile['name'],
        'knowledge_graph_enabled': student_kg is not None,
        'total_materials': len(materials),
        'generated_at': datetime.now().isoformat()
    })

# ============================================================================
# CHATBOT FUNCTIONALITY
# ============================================================================

class BITTutorChatbot:
    """Advanced AI Educational Chatbot for personalized learning assistance"""

    def __init__(self):
        self.conversation_history = {}
        self.knowledge_base = self._build_knowledge_base()
        self.response_templates = self._build_response_templates()

    def _build_knowledge_base(self):
        """Build comprehensive educational knowledge base"""
        return {
            'machine_learning': {
                'concepts': ['supervised learning', 'unsupervised learning', 'neural networks', 'deep learning', 'reinforcement learning'],
                'applications': ['computer vision', 'natural language processing', 'recommendation systems', 'autonomous vehicles'],
                'algorithms': ['linear regression', 'decision trees', 'random forest', 'svm', 'k-means', 'neural networks']
            },
            'data_science': {
                'concepts': ['data analysis', 'statistics', 'visualization', 'data cleaning', 'feature engineering'],
                'tools': ['python', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter'],
                'techniques': ['regression', 'classification', 'clustering', 'time series analysis']
            },
            'programming': {
                'languages': ['python', 'javascript', 'java', 'c++', 'r', 'sql'],
                'concepts': ['variables', 'functions', 'loops', 'conditionals', 'classes', 'objects'],
                'paradigms': ['object-oriented', 'functional', 'procedural']
            },
            'web_development': {
                'frontend': ['html', 'css', 'javascript', 'react', 'vue', 'angular'],
                'backend': ['node.js', 'python', 'flask', 'django', 'express'],
                'databases': ['mysql', 'postgresql', 'mongodb', 'sqlite']
            },
            'ai_ethics': {
                'topics': ['bias in ai', 'fairness', 'transparency', 'accountability', 'privacy'],
                'principles': ['beneficence', 'non-maleficence', 'autonomy', 'justice']
            }
        }

    def _build_response_templates(self):
        """Build response templates for different types of interactions"""
        return {
            'greeting': [
                "Hello! I'm your BIT Tutor learning assistant. How can I help you today?",
                "Hi there! Ready to explore some exciting learning topics?",
                "Welcome! I'm here to support your educational journey. What would you like to learn about?"
            ],
            'explanation': [
                "Great question! Let me explain {topic} in a way that connects to your interests.",
                "I'd be happy to break down {topic} for you, especially considering your background in {specialty}.",
                "Excellent topic choice! {topic} is fascinating, and I can relate it to your hobby of {hobby}."
            ],
            'encouragement': [
                "You're doing amazing! Your progress in {subject} shows real dedication.",
                "Keep up the excellent work! Your {streak} day learning streak is impressive.",
                "I can see you're really grasping these concepts. Your {engagement}% engagement rate is outstanding!"
            ],
            'suggestion': [
                "Based on your interest in {hobby}, you might enjoy learning about {related_topic}.",
                "Since you're working on {current_focus}, I recommend exploring {suggestion}.",
                "Given your {level} level, you're ready for more advanced topics like {advanced_topic}."
            ]
        }

    def get_student_context(self, student_id):
        """Get student context for personalized responses"""
        if student_id in students_data:
            return students_data[student_id]['profile']
        return None

    def analyze_message(self, message):
        """Analyze user message to determine intent and extract topics"""
        message_lower = message.lower()

        # Intent detection
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'start']):
            return {'intent': 'greeting', 'topics': []}
        elif any(word in message_lower for word in ['what', 'how', 'explain', 'tell me about']):
            return {'intent': 'question', 'topics': self._extract_topics(message_lower)}
        elif any(word in message_lower for word in ['help', 'stuck', 'confused', 'difficult']):
            return {'intent': 'help', 'topics': self._extract_topics(message_lower)}
        elif any(word in message_lower for word in ['recommend', 'suggest', 'what should', 'next']):
            return {'intent': 'recommendation', 'topics': self._extract_topics(message_lower)}
        else:
            return {'intent': 'general', 'topics': self._extract_topics(message_lower)}

    def _extract_topics(self, message):
        """Extract relevant topics from message"""
        topics = []
        for category, content in self.knowledge_base.items():
            for subcategory, items in content.items():
                for item in items:
                    if item.lower() in message:
                        topics.append({'category': category, 'topic': item})
        return topics

    def generate_response(self, message, student_id):
        """Generate personalized response based on message and student context"""
        student_context = self.get_student_context(student_id)
        analysis = self.analyze_message(message)

        # Initialize conversation history if needed
        if student_id not in self.conversation_history:
            self.conversation_history[student_id] = []

        # Add user message to history
        self.conversation_history[student_id].append({
            'type': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        # Generate response based on intent
        if analysis['intent'] == 'greeting':
            response = self._generate_greeting_response(student_context)
        elif analysis['intent'] == 'question':
            response = self._generate_explanation_response(analysis['topics'], student_context)
        elif analysis['intent'] == 'help':
            response = self._generate_help_response(analysis['topics'], student_context)
        elif analysis['intent'] == 'recommendation':
            response = self._generate_recommendation_response(student_context)
        else:
            response = self._generate_general_response(analysis['topics'], student_context)

        # Add bot response to history
        self.conversation_history[student_id].append({
            'type': 'bot',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })

        return response

    def _generate_greeting_response(self, student_context):
        """Generate personalized greeting"""
        if student_context:
            templates = [
                f"Hello {student_context['name']}! 👋 I see you're working on {student_context['current_focus']}. How can I assist you today?",
                f"Hi {student_context['name']}! 🌟 With your {student_context['streak_days']} day streak, you're on fire! What would you like to explore?",
                f"Welcome back {student_context['name']}! 🚀 Ready to dive deeper into {student_context['specialty']}?"
            ]
            return random.choice(templates)
        return random.choice(self.response_templates['greeting'])

    def _generate_explanation_response(self, topics, student_context):
        """Generate educational explanation"""
        if not topics:
            return "I'd be happy to explain any topic! Could you be more specific about what you'd like to learn?"

        topic = topics[0]
        category = topic['category']
        topic_name = topic['topic']

        explanations = {
            'machine_learning': {
                'supervised learning': "Supervised learning is like having a teacher guide you! 👨‍🏫 You provide the algorithm with input-output pairs (like showing it pictures of cats labeled 'cat'), and it learns to make predictions on new, unseen data.",
                'neural networks': "Neural networks are inspired by how our brain works! 🧠 They consist of interconnected nodes (neurons) that process information in layers, learning complex patterns from data.",
                'deep learning': "Deep learning uses neural networks with many layers (that's why it's 'deep'!) 🏗️ It's particularly powerful for tasks like image recognition and natural language processing."
            },
            'programming': {
                'python': "Python is like the Swiss Army knife of programming! 🐍 It's versatile, readable, and perfect for beginners. You can use it for web development, data science, AI, and more!",
                'functions': "Functions are like recipes in programming! 👨‍🍳 You define a set of instructions once, give it a name, and then you can 'call' it whenever you need those instructions executed.",
                'loops': "Loops are programming's way of saying 'repeat this!' 🔄 Instead of writing the same code multiple times, you use loops to automate repetitive tasks."
            },
            'data_science': {
                'data analysis': "Data analysis is like being a detective! 🕵️ You examine data to find patterns, trends, and insights that help answer questions or solve problems.",
                'visualization': "Data visualization turns numbers into pictures! 📊 It's much easier to understand trends and patterns when you can see them in charts and graphs."
            }
        }

        if category in explanations and topic_name in explanations[category]:
            base_explanation = explanations[category][topic_name]

            # Add personalized context if available
            if student_context:
                hobby_connections = self._get_hobby_connections(topic_name, student_context.get('hobbies', []))
                if hobby_connections:
                    base_explanation += f"\n\n🎯 **Connection to your interests**: {hobby_connections}"

            return base_explanation

        return f"Great question about {topic_name}! This is an important concept in {category}. Let me help you understand it better. Could you tell me what specific aspect you'd like me to focus on?"

    def _generate_help_response(self, topics, student_context):
        """Generate helpful response for struggling students"""
        if student_context:
            responses = [
                f"Don't worry {student_context['name']}, everyone faces challenges! 💪 With your {student_context['engagement_score']}% engagement, you're clearly dedicated. Let's break this down step by step.",
                f"I understand it can be frustrating, but you've got this! 🌟 Your {student_context['streak_days']} day streak shows your commitment. What specific part is giving you trouble?",
                f"Learning {student_context['specialty']} can be challenging, but that's how we grow! 🚀 Let me help you find a different approach that might click better."
            ]
            return random.choice(responses)

        return "I'm here to help! 🤝 Learning can be challenging, but every expert was once a beginner. What specific concept would you like me to explain differently?"

    def _generate_recommendation_response(self, student_context):
        """Generate personalized learning recommendations"""
        if not student_context:
            return "I'd love to give you personalized recommendations! Could you tell me about your current learning goals?"

        recommendations = []

        # Based on current focus
        focus_recommendations = {
            'Deep Learning Architectures': ['Convolutional Neural Networks', 'Transformer Models', 'GANs'],
            'Statistical Modeling': ['Bayesian Statistics', 'Time Series Analysis', 'A/B Testing'],
            'Object-Oriented Programming': ['Design Patterns', 'SOLID Principles', 'Unit Testing'],
            'Reinforcement Learning': ['Q-Learning', 'Policy Gradients', 'Multi-Agent Systems'],
            'System Design Patterns': ['Microservices', 'Event-Driven Architecture', 'Load Balancing']
        }

        current_focus = student_context.get('current_focus', '')
        if current_focus in focus_recommendations:
            next_topics = focus_recommendations[current_focus]
            recommendations.append(f"Since you're mastering {current_focus}, I recommend exploring: {', '.join(next_topics[:2])} 🎯")

        # Based on hobbies
        hobby_connections = []
        for hobby in student_context.get('hobbies', []):
            connection = self._get_hobby_learning_connection(hobby)
            if connection:
                hobby_connections.append(connection)

        if hobby_connections:
            recommendations.append(f"Based on your interests: {random.choice(hobby_connections)} 🎮")

        # Based on level
        level = student_context.get('level', 'Beginner')
        level_suggestions = {
            'Beginner': 'Try building a simple project to apply what you\'ve learned! 🏗️',
            'Intermediate': 'Consider contributing to an open-source project to gain real-world experience! 🌍',
            'Advanced': 'How about mentoring others or writing technical articles to solidify your expertise? ✍️',
            'Expert': 'Consider researching cutting-edge topics or starting your own innovative project! 🚀'
        }

        if level in level_suggestions:
            recommendations.append(level_suggestions[level])

        return '\n\n'.join(recommendations) if recommendations else "Keep up the great work! Your learning journey is progressing excellently! 🌟"

    def _generate_general_response(self, topics, student_context):
        """Generate general conversational response"""
        if topics:
            topic_names = [t['topic'] for t in topics]
            return f"Interesting topics you mentioned: {', '.join(topic_names)}! 🤔 I'd be happy to discuss any of these in detail. Which one would you like to explore first?"

        responses = [
            "I'm here to help with your learning journey! 📚 Feel free to ask me about any topic you're curious about.",
            "What's on your mind today? I can help explain concepts, suggest learning paths, or just chat about your interests! 💭",
            "I love helping students discover new things! What would you like to explore together? 🔍"
        ]

        return random.choice(responses)

    def _get_hobby_connections(self, topic, hobbies):
        """Connect learning topics to student hobbies"""
        connections = {
            'machine learning': {
                '🎮 Gaming': 'ML is used extensively in game AI for creating intelligent NPCs and procedural content generation!',
                '📸 Photography': 'Computer vision and ML power modern photo editing, object recognition, and automatic tagging!',
                '🎵 Music Production': 'AI can generate music, separate audio tracks, and even create personalized playlists!'
            },
            'neural networks': {
                '🎮 Gaming': 'Neural networks create adaptive game difficulty and realistic character behaviors!',
                '📸 Photography': 'CNNs (Convolutional Neural Networks) are the backbone of image recognition and enhancement!',
                '⚽ Soccer': 'Neural networks analyze player performance and predict game outcomes!'
            },
            'python': {
                '🎮 Gaming': 'Python is great for game development with libraries like Pygame!',
                '🍳 Cooking': 'You could build a recipe recommendation system or nutrition tracker!',
                '📚 Reading': 'Create text analysis tools or book recommendation engines!'
            }
        }

        for hobby in hobbies:
            if topic in connections and hobby in connections[topic]:
                return connections[topic][hobby]

        return None

    def _get_hobby_learning_connection(self, hobby):
        """Get learning suggestions based on hobbies"""
        suggestions = {
            '🎮 Gaming': 'Game development with Unity and C#, or AI for games',
            '📸 Photography': 'Computer vision and image processing with OpenCV',
            '🎵 Music Production': 'Audio signal processing and music information retrieval',
            '⚽ Soccer': 'Sports analytics and performance prediction models',
            '🎲 Board Games': 'Game theory and strategic AI algorithms',
            '🍳 Cooking': 'Recommendation systems and nutritional data analysis',
            '🎸 Guitar': 'Music theory analysis and chord progression algorithms',
            '🎬 Movies': 'Recommendation systems and sentiment analysis',
            '♟️ Chess': 'Advanced algorithms and game tree search',
            '🚀 Space Exploration': 'Orbital mechanics simulation and space data analysis'
        }

        return suggestions.get(hobby)

@app.route('/api/student/<student_id>/chat', methods=['POST'])
def chat_with_ai(student_id):
    """API endpoint for comprehensive AI chatbot interactions"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400

    user_message = data['message'].strip()
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    try:
        # Generate AI response using comprehensive system
        ai_response = comprehensive_ai.generate_comprehensive_response(user_message, student_id, students_data)

        # Get student context for response metadata
        profile = students_data[student_id]['profile']

        response_data = {
            'response': ai_response,
            'timestamp': datetime.now().isoformat(),
            'student_name': profile['name'],
            'conversation_id': str(uuid.uuid4()),
            'response_type': 'text',
            'suggestions': _generate_follow_up_suggestions(user_message, profile),
            'knowledge_graph_enabled': students_data[student_id].get('knowledge_graph') is not None
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

@app.route('/api/student/<student_id>/chat/history')
def get_chat_history(student_id):
    """API endpoint to get chat conversation history"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    history = comprehensive_ai.conversation_history.get(student_id, [])

    return jsonify({
        'history': history,
        'student_name': students_data[student_id]['profile']['name'],
        'total_messages': len(history),
        'knowledge_graph_enabled': students_data[student_id].get('knowledge_graph') is not None
    })

@app.route('/api/student/<student_id>/chat/clear', methods=['POST'])
def clear_chat_history(student_id):
    """API endpoint to clear chat history"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    if student_id in comprehensive_ai.conversation_history:
        comprehensive_ai.conversation_history[student_id] = []

    return jsonify({'message': 'Chat history cleared successfully'})

def _generate_follow_up_suggestions(user_message, profile):
    """Generate contextual follow-up suggestions"""
    suggestions = []
    message_lower = user_message.lower()

    # Context-based suggestions
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        suggestions = [
            f"Tell me about {profile['current_focus']}",
            "What study tips do you have for me?",
            "How can I connect my hobbies to learning?"
        ]
    elif any(word in message_lower for word in ['help', 'stuck', 'difficult']):
        suggestions = [
            "Can you break this down into smaller steps?",
            "What resources would you recommend?",
            "How do other students approach this?"
        ]
    elif any(word in message_lower for word in ['project', 'build', 'create', 'lab']):
        suggestions = [
            "What tools should I use for this project?",
            "How long would this project take?",
            "Can you suggest similar projects?"
        ]
    elif any(word in message_lower for word in ['quiz', 'test', 'assessment']):
        suggestions = [
            "What topics should I focus on?",
            "How can I prepare effectively?",
            "Are there practice questions available?"
        ]
    else:
        # Default suggestions based on profile
        suggestions = [
            f"Explain {profile['current_focus']} concepts",
            "Suggest practice exercises",
            "Connect this to my hobbies",
            "What should I learn next?"
        ]

    return suggestions[:3]  # Return top 3 suggestions

@app.route('/api/student/<student_id>/knowledge_graph/status')
def get_knowledge_graph_status(student_id):
    """API endpoint to show knowledge graph updates and status"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    try:
        student_kg = students_data[student_id].get('knowledge_graph')
        profile = students_data[student_id]['profile']

        if not student_kg:
            return jsonify({
                'status': 'not_initialized',
                'message': 'Knowledge graph not yet initialized for this student',
                'student_name': profile['name']
            })

        # Get current mastery levels
        mastery_levels = {}
        recent_updates = []

        if hasattr(student_kg, 'graph') and student_kg.graph:
            # Get mastery levels for all nodes
            for node in student_kg.graph.nodes():
                mastery = student_kg.graph.nodes[node].get('mastery_level', 0.0)
                mastery_levels[node] = mastery

            # Get recent updates (simulated for demo)
            recent_updates = [
                {
                    'concept': 'Machine Learning Basics',
                    'old_mastery': 0.6,
                    'new_mastery': 0.75,
                    'timestamp': datetime.now().isoformat(),
                    'trigger': 'Quiz completion'
                },
                {
                    'concept': 'Neural Networks',
                    'old_mastery': 0.3,
                    'new_mastery': 0.45,
                    'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                    'trigger': 'Lab exercise'
                },
                {
                    'concept': 'Data Preprocessing',
                    'old_mastery': 0.8,
                    'new_mastery': 0.85,
                    'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                    'trigger': 'Practice questions'
                }
            ]

        # Calculate overall progress
        total_concepts = len(mastery_levels) if mastery_levels else 10
        mastered_concepts = sum(1 for level in mastery_levels.values() if level >= 0.8) if mastery_levels else 0
        avg_mastery = sum(mastery_levels.values()) / len(mastery_levels) if mastery_levels else 0.0

        return jsonify({
            'status': 'active',
            'student_name': profile['name'],
            'knowledge_graph_stats': {
                'total_concepts': total_concepts,
                'mastered_concepts': mastered_concepts,
                'average_mastery': round(avg_mastery, 2),
                'mastery_percentage': round((mastered_concepts / total_concepts) * 100, 1) if total_concepts > 0 else 0
            },
            'mastery_levels': mastery_levels,
            'recent_updates': recent_updates,
            'update_triggers': [
                'Quiz completions',
                'Lab exercises',
                'Practice questions',
                'AI chat interactions',
                'Time spent on topics'
            ],
            'next_recommendations': [
                f"Focus on {profile['current_focus']} fundamentals",
                "Complete adaptive practice exercises",
                "Take diagnostic quiz for weak areas"
            ]
        })

    except Exception as e:
        return jsonify({'error': f'Failed to get knowledge graph status: {str(e)}'}), 500

@app.route('/api/student/<student_id>/knowledge_graph/visualization')
def get_knowledge_graph_visualization(student_id):
    """API endpoint to get knowledge graph visualization data"""
    if student_id not in students_data:
        return jsonify({'error': 'Student not found'}), 404

    try:
        student_kg = students_data[student_id].get('knowledge_graph')
        profile = students_data[student_id]['profile']

        if not student_kg:
            # Return sample visualization data
            return jsonify({
                'nodes': [
                    {'id': 'ml_basics', 'label': 'ML Basics', 'mastery': 0.75, 'category': 'foundation'},
                    {'id': 'neural_nets', 'label': 'Neural Networks', 'mastery': 0.45, 'category': 'advanced'},
                    {'id': 'data_prep', 'label': 'Data Preprocessing', 'mastery': 0.85, 'category': 'foundation'},
                    {'id': 'algorithms', 'label': 'Algorithms', 'mastery': 0.60, 'category': 'core'},
                    {'id': 'evaluation', 'label': 'Model Evaluation', 'mastery': 0.30, 'category': 'core'}
                ],
                'edges': [
                    {'source': 'ml_basics', 'target': 'neural_nets', 'strength': 0.8},
                    {'source': 'data_prep', 'target': 'ml_basics', 'strength': 0.9},
                    {'source': 'ml_basics', 'target': 'algorithms', 'strength': 0.7},
                    {'source': 'algorithms', 'target': 'evaluation', 'strength': 0.6}
                ],
                'student_name': profile['name'],
                'focus_area': profile['current_focus']
            })

        # If knowledge graph exists, extract real data
        nodes = []
        edges = []

        if hasattr(student_kg, 'graph') and student_kg.graph:
            for node in student_kg.graph.nodes():
                node_data = student_kg.graph.nodes[node]
                nodes.append({
                    'id': node,
                    'label': node.replace('_', ' ').title(),
                    'mastery': node_data.get('mastery_level', 0.0),
                    'category': node_data.get('category', 'general')
                })

            for edge in student_kg.graph.edges():
                edge_data = student_kg.graph.edges[edge]
                edges.append({
                    'source': edge[0],
                    'target': edge[1],
                    'strength': edge_data.get('weight', 0.5)
                })

        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'student_name': profile['name'],
            'focus_area': profile['current_focus']
        })

    except Exception as e:
        return jsonify({'error': f'Failed to get visualization data: {str(e)}'}), 500





if __name__ == '__main__':
    print("🌟 Starting BIT Tutor Educational Platform...")
    print("🔗 Access at: http://127.0.0.1:8080")
    app.run(debug=True, host='127.0.0.1', port=8080)
