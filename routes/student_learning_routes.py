# routes/student_learning_routes.py

"""
Student Learning Routes

This module provides routes for the comprehensive student learning interface:
1. Topic browsing and enrollment
2. Pre-topic assessments (mandatory before studying)
3. Learning materials (LLM-generated blogs)
4. Lab exercises
5. Quizzes and assessments
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import sys
from pathlib import Path

# Create Blueprint
student_learning_bp = Blueprint('student_learning', __name__, url_prefix='/student/<student_id>/learn')

# Initialize services (will be set by app)
graph_manager = None
assessment_engine = None
lab_tutor_loader = None
blog_generator = None
question_generator = None
quiz_generator = None
lab_generator = None
learning_assistant = None
content_fetcher = None


def init_student_learning_routes(app, dynamic_graph_manager, assessment_eng, lab_loader, blog_gen, question_gen, quiz_gen, lab_gen, assistant, fetcher=None):
    """
    Initialize student learning routes with the Flask app and services.

    Args:
        app: Flask application instance
        dynamic_graph_manager: Instance of DynamicGraphManager
        assessment_eng: Instance of AssessmentEngine
        lab_loader: Instance of LabTutorLoader
        blog_gen: Instance of LLMBlogGenerator
        question_gen: Instance of QuestionGenerator
        quiz_gen: Instance of QuizGenerator
        lab_gen: Instance of LabGenerator
        assistant: Instance of LearningAssistant
        fetcher: Instance of ContentFetcherAgent (optional)
    """
    global graph_manager, assessment_engine, lab_tutor_loader, blog_generator, question_generator, quiz_generator, lab_generator, learning_assistant, content_fetcher
    graph_manager = dynamic_graph_manager
    assessment_engine = assessment_eng
    content_fetcher = fetcher
    lab_tutor_loader = lab_loader
    blog_generator = blog_gen
    question_generator = question_gen
    quiz_generator = quiz_gen
    lab_generator = lab_gen
    learning_assistant = assistant
    app.register_blueprint(student_learning_bp)
    print("✅ Student learning routes initialized")


# ==================== TOPIC BROWSING ====================

@student_learning_bp.route('/topics')
def browse_topics(student_id):
    """
    Browse all available topics from Neo4j knowledge graph.
    """
    topics = []
    stats = {"total_topics": 0, "total_concepts": 0, "total_quizzes": 0, "total_labs": 0}

    if graph_manager and graph_manager.neo4j:
        try:
            # Get topics from the student's registered class
            query = """
            MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)-[:INCLUDES]->(t:Topic)
            OPTIONAL MATCH (t)-[:INCLUDES_CONCEPT]->(concept:Concept)
            OPTIONAL MATCH (quiz:Quiz)-[:TESTS]->(t)
            OPTIONAL MATCH (lab:Lab)-[:PRACTICES]->(t)
            WITH t,
                 count(DISTINCT concept) as concept_count,
                 count(DISTINCT quiz) as quiz_count,
                 count(DISTINCT lab) as lab_count
            RETURN t.topic_id as topic_id,
                   t.name as name,
                   t.description as description,
                   t.order as order,
                   t.estimated_hours as estimated_hours,
                   concept_count,
                   quiz_count,
                   lab_count
            ORDER BY t.order
            """
            result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})

            if result:
                for record in result:
                    topics.append({
                        'topic_id': record['topic_id'],
                        'name': record['name'],
                        'description': record['description'],
                        'order': record['order'],
                        'estimated_hours': record['estimated_hours'],
                        'concept_count': record['concept_count'],
                        'quiz_count': record['quiz_count'],
                        'lab_count': record['lab_count']
                    })
                    stats['total_concepts'] += record['concept_count']
                    stats['total_quizzes'] += record['quiz_count']
                    stats['total_labs'] += record['lab_count']

                stats['total_topics'] = len(topics)
        except Exception as e:
            print(f"Error fetching topics from Neo4j: {e}")

    # Get student's enrolled topics (all topics from their class)
    enrolled_topics = [t['name'] for t in topics]

    return render_template('student/topic_browser.html',
                         student_id=student_id,
                         topics=topics,
                         enrolled_topics=enrolled_topics,
                         stats=stats)


@student_learning_bp.route('/topics/<topic_name>')
def topic_detail(student_id, topic_name):
    """
    Comprehensive topic learning page with videos, readings, quiz, and lab.
    """
    # Get topic info from Neo4j
    topic_info = {
        "name": topic_name,
        "description": f"Learn about {topic_name}",
        "estimated_hours": 4
    }

    videos = []
    readings = []
    concepts = []
    quiz = None
    lab = None
    student_score = 0

    if graph_manager and graph_manager.neo4j:
        try:
            # Get topic details
            topic_query = """
            MATCH (t:Topic {name: $topic_name})
            RETURN t.description as description,
                   t.estimated_hours as estimated_hours
            """
            topic_result = graph_manager.neo4j.graph.query(topic_query, {'topic_name': topic_name})
            if topic_result:
                topic_info['description'] = topic_result[0].get('description', topic_info['description'])
                topic_info['estimated_hours'] = topic_result[0].get('estimated_hours', 4)

            # Get videos - first try from Neo4j, then fetch from online sources
            video_query = """
            MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)-[:EXPLAINED_BY]->(v:Video)
            RETURN v.title as title, v.url as url, v.duration as duration
            LIMIT 3
            """
            video_results = graph_manager.neo4j.graph.query(video_query, {'topic_name': topic_name})
            videos = [dict(v) for v in video_results] if video_results else []

            # If no videos in Neo4j and content_fetcher is available, fetch from online sources
            if not videos and content_fetcher:
                try:
                    videos = content_fetcher.fetch_videos_for_topic(topic_name, max_results=3)
                except Exception as e:
                    print(f"Error fetching videos from online sources: {e}")
                    videos = []

            # Get student hobbies and interests for personalization
            student_query = """
            MATCH (s:Student {student_id: $student_id})
            RETURN s.hobbies as hobbies, s.interests as interests
            """
            student_result = graph_manager.neo4j.graph.query(student_query, {'student_id': student_id})
            student_hobbies = []
            student_interests = []
            if student_result and len(student_result) > 0:
                student_hobbies = student_result[0].get('hobbies', [])
                student_interests = student_result[0].get('interests', [])

            # Get reading materials for ALL concepts with theory text for blog generation
            reading_query = """
            MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
            OPTIONAL MATCH (c)-[:EXPLAINED_BY]->(r:ReadingMaterial)
            OPTIONAL MATCH (th:Theory)-[:CONSISTS_OF]->(c)
            RETURN r.title as title, r.url as url, r.source as source,
                   c.name as concept_name, c.description as description,
                   th.compressed_text as theory_text,
                   th.keywords as keywords
            ORDER BY c.name, r.title
            """
            reading_results = graph_manager.neo4j.graph.query(reading_query, {'topic_name': topic_name})

            # Enhance readings with personalized blog content
            readings = []
            if reading_results:
                for r in reading_results:
                    reading = dict(r)
                    # Add personalization info
                    reading['student_hobbies'] = student_hobbies
                    reading['student_interests'] = student_interests
                    # Generate blog URL
                    reading['blog_url'] = url_for('student_learning.concept_reading_blog',
                                                  student_id=student_id,
                                                  topic_name=topic_name,
                                                  concept_name=reading['concept_name'])
                    readings.append(reading)

            # If no readings in Neo4j and content_fetcher is available, fetch from online sources
            if not readings and content_fetcher and concepts:
                try:
                    for concept in concepts[:5]:  # Fetch for first 5 concepts
                        fetched_readings = content_fetcher.fetch_reading_materials_for_concept(
                            concept['name'], topic_name, max_results=1
                        )
                        for reading in fetched_readings:
                            reading['student_hobbies'] = student_hobbies
                            reading['student_interests'] = student_interests
                            reading['blog_url'] = url_for('student_learning.concept_reading_blog',
                                                          student_id=student_id,
                                                          topic_name=topic_name,
                                                          concept_name=reading['concept_name'])
                            readings.append(reading)
                except Exception as e:
                    print(f"Error fetching reading materials from online sources: {e}")

            # Get concepts
            concept_query = """
            MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
            RETURN c.name as name, c.description as description
            ORDER BY c.name
            """
            concept_results = graph_manager.neo4j.graph.query(concept_query, {'topic_name': topic_name})
            concepts = [dict(c) for c in concept_results] if concept_results else []

            # Get ONE quiz for this topic
            quiz_query = """
            MATCH (q:Quiz)-[:TESTS]->(t:Topic {name: $topic_name})
            RETURN q.title as title, q.total_questions as total_questions
            LIMIT 1
            """
            quiz_results = graph_manager.neo4j.graph.query(quiz_query, {'topic_name': topic_name})
            if quiz_results:
                quiz = dict(quiz_results[0])

            # Get ONE comprehensive lab for this topic
            lab_query = """
            MATCH (l:Lab)-[:PRACTICES]->(t:Topic {name: $topic_name})
            RETURN l.lab_id as lab_id, l.title as title, l.objective as objective,
                   l.difficulty as difficulty, l.estimated_time as estimated_time
            LIMIT 1
            """
            lab_results = graph_manager.neo4j.graph.query(lab_query, {'topic_name': topic_name})
            if lab_results:
                lab = dict(lab_results[0])
            else:
                # If no lab exists, create a placeholder with lab_id
                lab = {
                    'lab_id': f"lab_{topic_name.lower().replace(' ', '_')}",
                    'title': f"Hands-On Lab: {topic_name}",
                    'objective': f"Practice the concepts learned in {topic_name}",
                    'difficulty': 'Intermediate',
                    'estimated_time': 90
                }

        except Exception as e:
            print(f"Error fetching topic data: {e}")

    # Render the tabbed learning page
    return render_template('student/topic_learning_tabbed.html',
                         student_id=student_id,
                         topic=topic_info,
                         videos=videos,
                         readings=readings,
                         concepts=concepts,
                         quiz=quiz,
                         lab=lab,
                         student_score=student_score)


@student_learning_bp.route('/api/topics/<topic_name>/refresh_videos', methods=['POST'])
def refresh_videos(student_id, topic_name):
    if not content_fetcher:
        return jsonify({"ok": False, "error": "Content fetcher not available"}), 400
    try:
        videos = content_fetcher.fetch_videos_for_topic(topic_name, max_results=3)
        return jsonify({"ok": True, "videos": videos})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@student_learning_bp.route('/api/topics/<topic_name>/refresh_readings', methods=['POST'])
def refresh_readings(student_id, topic_name):
    if not content_fetcher:
        return jsonify({"ok": False, "error": "Content fetcher not available"}), 400
    try:
        # Attempt to use first 5 concepts as anchors
        concept_query = """
        MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
        RETURN c.name as name
        ORDER BY c.name
        LIMIT 5
        """
        concept_results = graph_manager.neo4j.graph.query(concept_query, {'topic_name': topic_name}) if graph_manager and graph_manager.neo4j else []
        concepts = [r.get('name') for r in concept_results] if concept_results else []
        readings = []
        for c in concepts:
            fetched = content_fetcher.fetch_reading_materials_for_concept(c, topic_name, max_results=1)
            readings.extend(fetched)
        return jsonify({"ok": True, "readings": readings})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# Continue with existing routes below...


@student_learning_bp.route('/topics/<topic_name>/start', methods=['POST'])
def start_learning(student_id, topic_name):
    """
    Start learning a topic (redirects to pre-assessment).
    """
    if not graph_manager:
        return jsonify({"error": "Graph manager not available"}), 500

    # Check if topic exists
    if lab_tutor_loader:
        topic = lab_tutor_loader.get_topic_by_name(topic_name)
        if not topic:
            return jsonify({"error": "Topic not found"}), 404

    # Redirect to pre-topic assessment
    return jsonify({
        "success": True,
        "message": "Let's start with an assessment to personalize your learning",
        "redirect_url": url_for('student_learning.pre_topic_assessment',
                               student_id=student_id,
                               topic_name=topic_name)
    })


@student_learning_bp.route('/topics/<topic_name>/skip-assessment', methods=['POST'])
def skip_assessment(student_id, topic_name):
    """
    Skip the pre-assessment and go directly to learning materials.
    """
    if not graph_manager:
        return jsonify({"error": "Graph manager not available"}), 500

    # Check if topic exists
    if lab_tutor_loader:
        topic = lab_tutor_loader.get_topic_by_name(topic_name)
        if not topic:
            return jsonify({"error": "Topic not found"}), 404

    # Enroll student in topic (if not already enrolled)
    graph_manager.enroll_student_in_topic(student_id, topic_name)

    # Redirect to learning materials
    redirect_url = url_for('student_learning.learning_materials',
                          student_id=student_id,
                          topic_name=topic_name)

    return jsonify({"success": True, "redirect_url": redirect_url})


# ==================== PRE-TOPIC ASSESSMENT ====================

@student_learning_bp.route('/topics/<topic_name>/assessment')
def pre_topic_assessment(student_id, topic_name):
    """
    Display pre-topic assessment (optional - helps personalize learning).
    """
    if not lab_tutor_loader:
        return "Lab tutor loader not available", 500

    # Get topic and concepts
    topic_summary = lab_tutor_loader.get_topic_summary(topic_name)

    if not topic_summary.get('found'):
        return f"Topic '{topic_name}' not found", 404

    concepts = topic_summary.get('concepts', [])

    return render_template('student/pre_topic_assessment.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         concepts=concepts,
                         concept_count=len(concepts))


@student_learning_bp.route('/api/topics/<topic_name>/assessment/generate', methods=['POST'])
def generate_pre_assessment(student_id, topic_name):
    """
    API endpoint to generate pre-topic assessment questions.
    First tries to pull from existing Question nodes in Neo4j.
    If not found, generates new questions.
    """
    if not graph_manager:
        return jsonify({"error": "Graph manager not available"}), 500

    try:
        # First, try to get questions from Question nodes in Neo4j
        questions_query = """
        MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
        MATCH (q:Question)-[:TESTS]->(c)
        RETURN q.question_id as question_id,
               q.question as question,
               q.options as options,
               q.correct_answer as correct_answer,
               q.explanation as explanation,
               c.name as concept_name
        LIMIT 10
        """

        questions_result = graph_manager.neo4j.graph.query(
            questions_query,
            {'topic_name': topic_name}
        )

        all_questions = []

        if questions_result and len(questions_result) > 0:
            # Build assessment from existing Question nodes
            import json
            for record in questions_result:
                all_questions.append({
                    'question_id': record['question_id'],
                    'question': record['question'],
                    'options': json.loads(record['options']) if isinstance(record['options'], str) else record['options'],
                    'correct_answer': record['correct_answer'],
                    'explanation': record.get('explanation', ''),
                    'concept': record.get('concept_name', '')
                })
        else:
            # No questions found - generate new ones
            if not lab_tutor_loader or not question_generator:
                return jsonify({"error": "Cannot generate questions - services not available"}), 500

            # Get concepts and theories for this topic
            topic_summary = lab_tutor_loader.get_topic_summary(topic_name)
            concepts = topic_summary.get('concepts', [])
            theories = topic_summary.get('theories', [])

            if not concepts:
                return jsonify({"error": "No concepts found for this topic"}), 404

            # Generate questions for each concept
            selected_concepts = concepts[:10] if len(concepts) > 10 else concepts  # Max 10 concepts

            for concept in selected_concepts:
                # Find theory data for this concept
                theory_data = None
                for theory in theories:
                    if concept in theory.get('keywords', []):
                        theory_data = theory
                        break

                # Generate 1 question per concept for pre-assessment
                questions = question_generator.generate_questions_for_concept(
                    concept_name=concept,
                    topic_name=topic_name,
                    theory_data=theory_data,
                    num_questions=1
                )

                all_questions.extend(questions)

            # Store questions in graph
            if graph_manager and all_questions:
                question_generator.store_questions_in_graph(all_questions, graph_manager)

        # Format for assessment
        assessment_data = {
            'assessment_id': f"pre_{topic_name}_{student_id}_{datetime.now().timestamp()}",
            'topic_name': topic_name,
            'questions': all_questions,
            'total_questions': len(all_questions),
            'time_limit': len(all_questions) * 60  # 1 minute per question
        }

        return jsonify(assessment_data)
    except Exception as e:
        print(f"Error generating assessment: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@student_learning_bp.route('/api/topics/<topic_name>/assessment/submit', methods=['POST'])
def submit_pre_assessment(student_id, topic_name):
    """
    API endpoint to submit pre-topic assessment.
    """
    if not assessment_engine or not graph_manager:
        return jsonify({"error": "Services not available"}), 500

    data = request.get_json()
    assessment_id = data.get('assessment_id')
    answers = data.get('answers', {})
    time_taken = data.get('time_taken', 0)

    try:
        # Evaluate assessment
        results = assessment_engine.evaluate_assessment(
            assessment_id=assessment_id,
            student_answers=answers,
            time_taken=time_taken
        )

        # Record in graph
        if graph_manager:
            graph_manager.record_assessment_attempt(
                student_id=student_id,
                topic_name=topic_name,
                assessment_type="pre-topic",
                score=results['overall_score'],
                concept_scores=results['concept_scores']
            )

            # Mark as started (no enrollment needed)
            graph_manager.enroll_student_in_topic(student_id, topic_name)

        # Generate personalized learning path based on results
        personalized_concepts = _create_personalized_learning_path(
            student_id,
            topic_name,
            results.get('concept_scores', {})
        )

        results['personalized_concepts'] = personalized_concepts
        results['message'] = "Assessment complete! Your personalized learning path has been created."

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== LEARNING MATERIALS ====================

@student_learning_bp.route('/topics/<topic_name>/materials')
def learning_materials(student_id, topic_name):
    """
    Display learning materials for a topic.
    Shows personalized concepts based on assessment results (if available).
    Auto-enrolls student if not already enrolled.
    """
    # Auto-enroll student if not enrolled
    personalized_concepts = []
    if graph_manager:
        portfolio = graph_manager.get_student_portfolio(student_id)
        is_enrolled = topic_name in portfolio.get('topic_progress', {})
        if not is_enrolled:
            # Auto-enroll the student
            graph_manager.enroll_student_in_topic(student_id, topic_name)

        # Get personalized learning path (if assessment was taken)
        personalized_concepts = _get_personalized_concepts(student_id, topic_name, portfolio)

    # Get concepts for this topic
    if lab_tutor_loader:
        topic_summary = lab_tutor_loader.get_topic_summary(topic_name)
        concepts = topic_summary.get('concepts', [])
    else:
        concepts = []

    return render_template('student/learning_materials.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         concepts=concepts,
                         personalized_concepts=personalized_concepts)


@student_learning_bp.route('/topics/<topic_name>/materials/<concept_name>')
def concept_material(student_id, topic_name, concept_name):
    """
    Display learning material (blog) for a specific concept.
    """
    if not blog_generator or not lab_tutor_loader:
        return "Services not available", 500

    # Get theory data for this concept
    topic_summary = lab_tutor_loader.get_topic_summary(topic_name)
    theories = topic_summary.get('theories', [])

    # Find theory related to this concept
    theory_data = None
    for theory in theories:
        if concept_name in theory.get('keywords', []):
            theory_data = theory
            break

    # Get student profile for personalization
    student_level = "intermediate"
    learning_style = "visual"
    if graph_manager:
        portfolio = graph_manager.get_student_portfolio(student_id)
        student_level = portfolio.get('level', 'intermediate').lower()
        learning_style = portfolio.get('learning_style', 'visual').lower()

    # Generate blog
    blog_data = blog_generator.generate_blog(
        concept_name=concept_name,
        topic=topic_name,
        theory_data=theory_data,
        student_level=student_level,
        learning_style=learning_style
    )

    return render_template('student/learning_material.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         concept_name=concept_name,
                         blog=blog_data)


@student_learning_bp.route('/api/topics/<topic_name>/materials/<concept_name>/complete', methods=['POST'])
def complete_material(student_id, topic_name, concept_name):
    """
    Mark a learning material as completed.
    """
    if not graph_manager:
        return jsonify({"error": "Graph manager not available"}), 500

    data = request.get_json()
    time_spent = data.get('time_spent', 0)

    try:
        # Record completion
        graph_manager.record_material_completion(
            student_id=student_id,
            topic_name=topic_name,
            concept_name=concept_name,
            time_spent=time_spent
        )

        return jsonify({"success": True, "message": "Material marked as completed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== LABS ====================

@student_learning_bp.route('/topics/<topic_name>/labs')
def topic_labs(student_id, topic_name):
    """
    Display available labs for a topic (practical exercises).
    Pull labs from Neo4j knowledge graph.
    """
    if not graph_manager:
        return "Graph manager not available", 500

    # Get labs from Neo4j
    labs_query = """
    MATCH (l:Lab)-[:PRACTICES]->(t:Topic {name: $topic_name})
    RETURN l.lab_id as lab_id,
           l.title as title,
           l.objective as objective,
           l.difficulty as difficulty,
           l.estimated_time as estimated_time,
           l.instructions as instructions,
           l.concepts_covered as concepts_covered
    ORDER BY
        CASE l.difficulty
            WHEN 'beginner' THEN 1
            WHEN 'intermediate' THEN 2
            WHEN 'advanced' THEN 3
            ELSE 4
        END
    """

    try:
        result = graph_manager.neo4j.graph.query(
            labs_query,
            {'topic_name': topic_name}
        )
    except Exception as e:
        print(f"Error fetching labs from Neo4j: {e}")
        result = None

    if not result:
        # Fallback to generating labs if not in Neo4j
        if not lab_tutor_loader or not lab_generator:
            return "Labs not found and cannot generate", 404

        topic_summary = lab_tutor_loader.get_topic_summary(topic_name)
        concepts = topic_summary.get('concepts', [])
        theories = topic_summary.get('theories', [])

        labs = lab_generator.generate_labs_for_topic(
            topic_name=topic_name,
            concepts=concepts,
            theory_data=theories,
            num_labs=3
        )
    else:
        # Parse labs from Neo4j
        import json
        labs = []
        for lab_record in result:
            lab = {
                'lab_id': lab_record['lab_id'],
                'name': lab_record['title'],
                'title': lab_record['title'],
                'description': lab_record['objective'],
                'objective': lab_record['objective'],
                'difficulty': lab_record['difficulty'],
                'estimated_time': lab_record['estimated_time'],
                'instructions': json.loads(lab_record['instructions']) if isinstance(lab_record['instructions'], str) else lab_record['instructions'],
                'concepts_covered': json.loads(lab_record['concepts_covered']) if isinstance(lab_record['concepts_covered'], str) else lab_record['concepts_covered'],
                'completed': False  # TODO: Check student progress
            }
            labs.append(lab)

    return render_template('student/topic_labs.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         labs=labs)


@student_learning_bp.route('/topics/<topic_name>/labs/<lab_id>')
def lab_notebook(student_id, topic_name, lab_id):
    """
    Display interactive Jupyter-style notebook for a specific lab.
    Content based on topic theory and concepts, background based on student hobbies.
    """
    if not graph_manager:
        return "Graph manager not available", 500

    # Get student hobby for background customization
    student_hobby = "learning"
    try:
        student_query = """
        MATCH (s:Student {student_id: $student_id})
        RETURN s.hobbies as hobbies
        """
        student_result = graph_manager.neo4j.graph.query(student_query, {'student_id': student_id})
        if student_result and student_result[0]['hobbies']:
            hobbies = student_result[0]['hobbies']
            student_hobby = hobbies[0] if isinstance(hobbies, list) else hobbies
    except Exception as e:
        print(f"Error fetching student hobbies: {e}")

    # Get lab from Neo4j
    lab_query = """
    MATCH (l:Lab {lab_id: $lab_id})-[:PRACTICES]->(t:Topic {name: $topic_name})
    RETURN l.lab_id as lab_id,
           l.title as title,
           l.objective as objective,
           l.difficulty as difficulty,
           l.estimated_time as estimated_time,
           l.instructions as instructions,
           l.concepts_covered as concepts_covered,
           l.cells as cells
    """

    try:
        result = graph_manager.neo4j.graph.query(
            lab_query,
            {'lab_id': lab_id, 'topic_name': topic_name}
        )

        if not result:
            return "Lab not found", 404

        import json
        lab_data = result[0]

        # Get concepts for theory section
        concepts_query = """
        MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
        RETURN c.name as name, c.description as description, c.example as example
        LIMIT 5
        """
        concepts_result = graph_manager.neo4j.graph.query(concepts_query, {'topic_name': topic_name})
        concepts = [dict(c) for c in concepts_result] if concepts_result else []

        # Parse lab cells
        cells = json.loads(lab_data['cells']) if isinstance(lab_data['cells'], str) else lab_data['cells']
        if not cells:
            # Generate default cells based on concepts
            cells = [
                {
                    'title': 'Import Libraries',
                    'instructions': 'Import the necessary Python libraries for this lab.',
                    'code': '# Import required libraries\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt',
                    'hint': 'Make sure all libraries are installed in your environment.'
                },
                {
                    'title': 'Load Data',
                    'instructions': 'Load and explore the dataset.',
                    'code': '# Load your data here\n# data = pd.read_csv("data.csv")\n# print(data.head())',
                    'hint': 'Use pandas read_csv() function to load CSV files.'
                },
                {
                    'title': 'Your Solution',
                    'instructions': 'Implement the main logic based on the concepts learned.',
                    'code': '# Write your code here\n',
                    'hint': 'Review the theory section above for guidance.'
                }
            ]

        lab = {
            'lab_id': lab_data['lab_id'],
            'title': lab_data['title'],
            'objective': lab_data['objective'],
            'difficulty': lab_data['difficulty'],
            'estimated_time': lab_data['estimated_time'],
            'concepts': concepts,
            'cells': cells
        }

    except Exception as e:
        print(f"Error fetching lab: {e}")
        import traceback
        traceback.print_exc()
        return "Error loading lab", 500

    return render_template('student/lab_notebook.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         lab=lab,
                         student_hobby=student_hobby)


@student_learning_bp.route('/api/execute-code', methods=['POST'])
def execute_code(student_id):
    """
    Execute Python code from lab notebook cells.
    Uses a sandboxed environment for safety.
    """
    data = request.get_json()
    code = data.get('code', '')

    if not code:
        return jsonify({'success': False, 'error': 'No code provided'}), 400

    try:
        # Simple execution using exec (in production, use a proper sandbox like RestrictedPython)
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        # Capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        # Create a restricted namespace
        namespace = {
            '__builtins__': __builtins__,
            'print': print,
            'range': range,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
        }

        # Try to import common data science libraries
        try:
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            namespace['np'] = np
            namespace['pd'] = pd
            namespace['plt'] = plt
        except ImportError:
            pass

        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, namespace)

        output = output_buffer.getvalue()
        error = error_buffer.getvalue()

        return jsonify({
            'success': True,
            'output': output,
            'error': error if error else None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@student_learning_bp.route('/api/submit-lab', methods=['POST'])
def submit_lab(student_id):
    """
    Submit completed lab and record progress.
    """
    data = request.get_json()
    topic_name = data.get('topic_name')
    lab_id = data.get('lab_id')
    cells = data.get('cells', [])

    if not topic_name or not lab_id:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    try:
        # Calculate score based on cell completion
        completed_cells = sum(1 for cell in cells if cell.get('code', '').strip())
        total_cells = len(cells)
        score = (completed_cells / total_cells * 100) if total_cells > 0 else 0

        # Record lab completion in Neo4j and add grade to student attributes
        if graph_manager:
            from datetime import datetime
            import json

            # Get current grades
            get_grades_query = """
            MATCH (s:Student {student_id: $student_id})
            OPTIONAL MATCH (l:Lab {lab_id: $lab_id})-[:PRACTICES]->(t:Topic)
            RETURN s.grades as grades_json, t.name as topic_name
            """
            result = graph_manager.neo4j.graph.query(get_grades_query, {
                'student_id': student_id,
                'lab_id': lab_id
            })

            grades = []
            topic_name = 'Lab Exercise'

            if result and len(result) > 0:
                grades_json = result[0].get('grades_json')
                topic_name = result[0].get('topic_name', 'Lab Exercise')
                if grades_json:
                    try:
                        grades = json.loads(grades_json)
                    except:
                        grades = []

            # Add new grade
            new_grade = {
                'grade_id': f'lab_{student_id}_{lab_id}_{int(datetime.now().timestamp())}',
                'type': 'lab',
                'score': score,
                'max_score': 100.0,
                'percentage': score,
                'date': datetime.now().isoformat(),
                'topic_name': topic_name
            }
            grades.append(new_grade)

            # Update student with new grades
            update_query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (l:Lab {lab_id: $lab_id})
            MERGE (s)-[c:COMPLETED]->(l)
            SET c.completed_at = datetime(),
                c.score = $score,
                c.cells_completed = $completed_cells,
                c.total_cells = $total_cells,
                s.grades = $grades_json,
                s.total_grades = $total_grades,
                s.last_updated = datetime()
            RETURN s.student_id
            """
            graph_manager.neo4j.graph.query(update_query, {
                'student_id': student_id,
                'lab_id': lab_id,
                'score': score,
                'completed_cells': completed_cells,
                'total_cells': total_cells,
                'grades_json': json.dumps(grades),
                'total_grades': len(grades)
            })

        return jsonify({
            'success': True,
            'score': score,
            'message': 'Lab submitted successfully!'
        })

    except Exception as e:
        print(f"Error submitting lab: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== CLASS PRE-ASSESSMENT ====================

@student_learning_bp.route('/class-assessment')
def class_pre_assessment(student_id):
    """
    Display class-level pre-assessment that students must take before accessing topics.
    """
    if not graph_manager:
        return "Graph manager not available", 500

    # Get student's class
    class_query = """
    MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)
    RETURN c.class_id as class_id, c.name as class_name, c.description as description
    """

    try:
        result = graph_manager.neo4j.graph.query(class_query, {'student_id': student_id})
        if not result:
            return "Student not registered in any class", 404

        class_data = result[0]

        # Check if already completed
        check_query = """
        MATCH (s:Student {student_id: $student_id})
        RETURN s.class_assessment_completed as completed
        """
        check_result = graph_manager.neo4j.graph.query(check_query, {'student_id': student_id})

        if check_result and check_result[0].get('completed'):
            # Already completed, redirect to topics
            return redirect(f'/student/{student_id}/learn/topics')

        return render_template('student/class_pre_assessment.html',
                             student_id=student_id,
                             class_id=class_data['class_id'],
                             class_name=class_data['class_name'],
                             class_description=class_data.get('description', ''))

    except Exception as e:
        print(f"Error loading class assessment: {e}")
        import traceback
        traceback.print_exc()
        return "Error loading assessment", 500


def generate_question_from_concept(concept):
    """
    Dynamically generate a question from a concept.
    """
    import random
    import uuid

    concept_name = concept['name']
    concept_desc = concept.get('description', '')
    topic_name = concept['topic_name']

    # Question templates based on concept type
    templates = [
        {
            'question': f"What is the primary purpose of {concept_name}?",
            'options': [
                f"To implement {concept_name} in databases",
                f"To understand {concept_name} concepts",
                f"To optimize {concept_name} performance",
                f"To design {concept_name} systems"
            ],
            'correct': 1
        },
        {
            'question': f"Which of the following best describes {concept_name}?",
            'options': [
                f"{concept_desc[:100] if concept_desc else 'A fundamental concept in ' + topic_name}",
                f"An advanced technique unrelated to {topic_name}",
                f"A deprecated approach in modern systems",
                f"A proprietary technology"
            ],
            'correct': 0
        },
        {
            'question': f"In the context of {topic_name}, {concept_name} is used to:",
            'options': [
                f"Improve system performance and scalability",
                f"Replace traditional database systems",
                f"Eliminate the need for data modeling",
                f"Reduce development costs"
            ],
            'correct': 0
        },
        {
            'question': f"What is a key characteristic of {concept_name}?",
            'options': [
                f"It is essential for understanding {topic_name}",
                f"It is only used in legacy systems",
                f"It requires no prior knowledge",
                f"It is independent of database design"
            ],
            'correct': 0
        }
    ]

    # Select a random template
    template = random.choice(templates)

    return {
        'question_id': f"gen_{uuid.uuid4().hex[:12]}",
        'concept_id': concept['concept_id'],
        'concept_name': concept_name,
        'topic_name': topic_name,
        'question': template['question'],
        'options': template['options'],
        'correct_answer': template['correct'],
        'difficulty': 'medium'
    }


@student_learning_bp.route('/api/class-assessment/generate', methods=['POST'])
def generate_class_assessment(student_id):
    """
    Generate class-level pre-assessment covering all concepts in the class.
    """
    if not graph_manager:
        return jsonify({"error": "Graph manager not available"}), 500

    data = request.get_json()
    class_id = data.get('class_id')

    try:
        # Get all concepts from all topics in the class
        concepts_query = """
        MATCH (c:Class {class_id: $class_id})-[:INCLUDES]->(t:Topic)-[:INCLUDES_CONCEPT]->(con:Concept)
        RETURN DISTINCT con.concept_id as concept_id,
               con.name as name,
               con.description as description,
               t.name as topic_name
        """

        concepts_result = graph_manager.neo4j.graph.query(concepts_query, {'class_id': class_id})

        if not concepts_result:
            return jsonify({"error": "No concepts found for this class"}), 404

        # Generate questions for each concept
        all_questions = []
        import random
        import json

        for concept in concepts_result:
            # Try to get questions from Neo4j first
            questions_query = """
            MATCH (q:Question)-[:TESTS]->(c:Concept {concept_id: $concept_id})
            RETURN q.question_id as question_id,
                   q.question as question,
                   q.options as options,
                   q.correct_answer as correct_answer,
                   q.difficulty as difficulty
            LIMIT 2
            """

            questions_result = graph_manager.neo4j.graph.query(
                questions_query,
                {'concept_id': concept['concept_id']}
            )

            if questions_result and len(questions_result) > 0:
                # Use existing questions from Neo4j
                for q in questions_result:
                    all_questions.append({
                        'question_id': q['question_id'],
                        'concept_id': concept['concept_id'],
                        'concept_name': concept['name'],
                        'topic_name': concept['topic_name'],
                        'question': q['question'],
                        'options': json.loads(q['options']) if isinstance(q['options'], str) else q['options'],
                        'correct_answer': q['correct_answer'],
                        'difficulty': q.get('difficulty', 'medium')
                    })
            else:
                # Dynamically generate questions based on concept
                generated_question = generate_question_from_concept(concept)
                if generated_question:
                    all_questions.append(generated_question)

        # Limit to 20 questions for the assessment
        selected_questions = random.sample(all_questions, min(20, len(all_questions))) if all_questions else []

        assessment_id = f"class_pre_{student_id}_{class_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return jsonify({
            'assessment_id': assessment_id,
            'class_id': class_id,
            'questions': selected_questions,
            'total_questions': len(selected_questions),
            'time_limit': len(selected_questions) * 90  # 1.5 minutes per question
        })

    except Exception as e:
        print(f"Error generating class assessment: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@student_learning_bp.route('/api/class-assessment/submit', methods=['POST'])
def submit_class_assessment(student_id):
    """
    Submit class pre-assessment and update student's cognitive profile.
    """
    if not graph_manager:
        return jsonify({"error": "Graph manager not available"}), 500

    data = request.get_json()
    class_id = data.get('class_id')
    assessment_id = data.get('assessment_id')
    responses = data.get('responses', [])
    time_spent = data.get('time_spent', 0)

    try:
        # Calculate score
        correct_count = sum(1 for r in responses if r.get('correct'))
        total_questions = len(responses)
        score = (correct_count / total_questions * 100) if total_questions > 0 else 0

        # Calculate mastery per concept
        concept_mastery = {}
        for response in responses:
            concept_id = response.get('concept_id')
            if concept_id not in concept_mastery:
                concept_mastery[concept_id] = {'correct': 0, 'total': 0, 'name': ''}

            concept_mastery[concept_id]['total'] += 1
            if response.get('correct'):
                concept_mastery[concept_id]['correct'] += 1

        # Get concept names and calculate mastery levels
        mastery_profile = []
        for concept_id, stats in concept_mastery.items():
            mastery_level = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

            # Get concept name
            concept_query = """
            MATCH (c:Concept {concept_id: $concept_id})
            RETURN c.name as name
            """
            concept_result = graph_manager.neo4j.graph.query(concept_query, {'concept_id': concept_id})
            concept_name = concept_result[0]['name'] if concept_result else concept_id

            mastery_profile.append({
                'concept_id': concept_id,
                'concept_name': concept_name,
                'mastery_level': mastery_level,
                'correct': stats['correct'],
                'total': stats['total']
            })

            # Update student-concept mastery in Neo4j
            update_mastery_query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (c:Concept {concept_id: $concept_id})
            MERGE (s)-[k:KNOWS]->(c)
            SET k.mastery_level = $mastery_level,
                k.last_assessed = datetime(),
                k.assessment_type = 'class_pre_assessment'
            """
            graph_manager.neo4j.graph.query(update_mastery_query, {
                'student_id': student_id,
                'concept_id': concept_id,
                'mastery_level': mastery_level
            })

        # Mark assessment as completed
        complete_query = """
        MATCH (s:Student {student_id: $student_id})
        SET s.class_assessment_completed = true,
            s.class_assessment_score = $score,
            s.class_assessment_date = datetime()
        """
        graph_manager.neo4j.graph.query(complete_query, {
            'student_id': student_id,
            'score': score
        })

        # Generate recommendations based on weak areas
        weak_concepts = [m for m in mastery_profile if m['mastery_level'] < 0.5]
        recommendations = []

        for weak in weak_concepts[:5]:  # Top 5 weak areas
            # Find topic for this concept
            topic_query = """
            MATCH (t:Topic)-[:INCLUDES_CONCEPT]->(c:Concept {concept_id: $concept_id})
            RETURN t.name as topic_name
            """
            topic_result = graph_manager.neo4j.graph.query(topic_query, {'concept_id': weak['concept_id']})

            if topic_result:
                recommendations.append({
                    'topic': topic_result[0]['topic_name'],
                    'concept': weak['concept_name'],
                    'reason': f"Low mastery ({weak['mastery_level']*100:.0f}%) - Start here to build foundation"
                })

        return jsonify({
            'success': True,
            'score': score,
            'correct_count': correct_count,
            'total_questions': total_questions,
            'mastery_profile': sorted(mastery_profile, key=lambda x: x['mastery_level']),
            'recommendations': recommendations,
            'time_spent': time_spent
        })

    except Exception as e:
        print(f"Error submitting class assessment: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==================== QUIZZES ====================

@student_learning_bp.route('/api/quiz/generate', methods=['POST'])
def generate_quiz_api(student_id):
    """
    API endpoint to generate a quiz dynamically for a topic.
    Called by the graded_quiz.html page.
    """
    if not graph_manager:
        return jsonify({'error': 'Graph manager not available'}), 500

    data = request.get_json()
    topic_name = data.get('topic_name')
    num_questions = data.get('num_questions', 15)

    if not topic_name:
        return jsonify({'error': 'Topic name is required'}), 400

    try:
        # First, try to get questions from Question nodes in Neo4j
        questions_query = """
        MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
        MATCH (q:Question)-[:TESTS]->(c)
        RETURN q.question_id as question_id,
               q.question as question,
               q.options as options,
               q.correct_answer as correct_answer,
               q.explanation as explanation,
               c.name as concept_name
        LIMIT $num_questions
        """

        questions_result = graph_manager.neo4j.graph.query(
            questions_query,
            {'topic_name': topic_name, 'num_questions': num_questions}
        )

        if questions_result and len(questions_result) > 0:
            # Build quiz from Question nodes
            import json
            questions = []
            for record in questions_result:
                questions.append({
                    'question_id': record['question_id'],
                    'question': record['question'],
                    'options': json.loads(record['options']) if isinstance(record['options'], str) else record['options'],
                    'correct_answer': record['correct_answer'],
                    'explanation': record.get('explanation', ''),
                    'concept': record.get('concept_name', '')
                })

            quiz_data = {
                'quiz_id': f"quiz_{topic_name}_{datetime.now().timestamp()}",
                'title': f"{topic_name} - Quiz",
                'questions': questions,
                'total_questions': len(questions),
                'passing_score': 0.7,
                'time_limit': len(questions) * 60  # 1 minute per question
            }

            return jsonify({
                'success': True,
                'quiz': quiz_data
            })
        else:
            # Fallback: Generate quiz using quiz_generator
            if not quiz_generator:
                return jsonify({'error': 'Quiz not found and cannot generate'}), 404

            # Fetch concepts from Neo4j
            concept_query = """
            MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
            RETURN c.name as name, c.description as description
            ORDER BY c.name
            """
            concept_results = graph_manager.neo4j.graph.query(concept_query, {'topic_name': topic_name})
            concepts = [c['name'] for c in concept_results] if concept_results else []

            # Fetch theory data
            theory_query = """
            MATCH (t:Topic {name: $topic_name})-[:HAS_THEORY]->(th:Theory)
            RETURN th.name as name, th.compressed_text as text
            """
            theory_results = graph_manager.neo4j.graph.query(theory_query, {'topic_name': topic_name})
            theories = [{'name': t['name'], 'text': t.get('text', '')} for t in theory_results] if theory_results else []

            quiz_data = quiz_generator.generate_quiz_for_topic(
                topic_name=topic_name,
                concepts=concepts,
                theory_data=theories,
                num_questions=num_questions
            )

            return jsonify({
                'success': True,
                'quiz': quiz_data
            })

    except Exception as e:
        print(f"Error generating quiz: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@student_learning_bp.route('/topics/<topic_name>/quiz')
def topic_quiz(student_id, topic_name):
    """
    Display quiz for a topic (end-of-chapter graded quiz).
    Pull questions from Neo4j Question nodes generated by agent.
    """
    if not graph_manager:
        return "Graph manager not available", 500

    # First, try to get questions from Question nodes in Neo4j
    questions_query = """
    MATCH (t:Topic {name: $topic_name})-[:INCLUDES_CONCEPT]->(c:Concept)
    MATCH (q:Question)-[:TESTS]->(c)
    RETURN q.question_id as question_id,
           q.question as question,
           q.options as options,
           q.correct_answer as correct_answer,
           q.explanation as explanation,
           c.name as concept_name
    LIMIT 15
    """

    try:
        questions_result = graph_manager.neo4j.graph.query(
            questions_query,
            {'topic_name': topic_name}
        )

        if questions_result and len(questions_result) > 0:
            # Build quiz from Question nodes
            import json
            questions = []
            for record in questions_result:
                questions.append({
                    'question_id': record['question_id'],
                    'question': record['question'],
                    'options': json.loads(record['options']) if isinstance(record['options'], str) else record['options'],
                    'correct_answer': record['correct_answer'],
                    'explanation': record.get('explanation', ''),
                    'concept': record.get('concept_name', '')
                })

            quiz_data = {
                'quiz_id': f"quiz_{topic_name}_{datetime.now().timestamp()}",
                'title': f"{topic_name} - Quiz",
                'questions': questions,
                'total_questions': len(questions),
                'passing_score': 0.7,
                'time_limit': len(questions) * 60  # 1 minute per question
            }
        else:
            # Fallback: Try to get quiz from Quiz node
            quiz_query = """
            MATCH (q:Quiz)-[:TESTS]->(t:Topic {name: $topic_name})
            RETURN q.quiz_id as quiz_id,
                   q.title as title,
                   q.questions as questions,
                   q.total_questions as total_questions,
                   q.passing_score as passing_score,
                   q.time_limit as time_limit
            LIMIT 1
            """

            quiz_result = graph_manager.neo4j.graph.query(quiz_query, {'topic_name': topic_name})

            if quiz_result:
                # Parse quiz from Neo4j
                import json
                quiz_record = quiz_result[0]
                quiz_data = {
                    'quiz_id': quiz_record['quiz_id'],
                    'title': quiz_record['title'],
                    'questions': json.loads(quiz_record['questions']) if isinstance(quiz_record['questions'], str) else quiz_record['questions'],
                    'total_questions': quiz_record['total_questions'],
                    'passing_score': quiz_record['passing_score'],
                    'time_limit': quiz_record['time_limit']
                }
            else:
                # Last resort: Generate quiz
                if not lab_tutor_loader or not quiz_generator:
                    return "Quiz not found and cannot generate", 404

                topic_summary = lab_tutor_loader.get_topic_summary(topic_name)
                concepts = topic_summary.get('concepts', [])
                theories = topic_summary.get('theories', [])

                quiz_data = quiz_generator.generate_quiz_for_topic(
                    topic_name=topic_name,
                    concepts=concepts,
                    theory_data=theories,
                    num_questions=15
                )

    except Exception as e:
        print(f"Error fetching quiz from Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return f"Error loading quiz: {str(e)}", 500

    return render_template('student/graded_quiz.html',
                         student_id=student_id,
                         topic_name=topic_name,
                         quiz=quiz_data)


# ==================== CHATBOT ====================

@student_learning_bp.route('/api/chatbot/message', methods=['POST'])
def chatbot_message(student_id):
    """
    API endpoint for chatbot messages.
    """
    if not learning_assistant:
        return jsonify({"error": "Learning assistant not available"}), 500

    data = request.get_json()
    message = data.get('message', '')
    context = data.get('context', {})

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = learning_assistant.get_response(student_id, message, context)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@student_learning_bp.route('/api/chatbot/history')
def chatbot_history(student_id):
    """
    Get chatbot conversation history.
    """
    if not learning_assistant:
        return jsonify({"error": "Learning assistant not available"}), 500

    try:
        history = learning_assistant.get_conversation_history(student_id)
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== HELPER FUNCTIONS ====================

def _get_personalized_concepts(student_id: str, topic_name: str, portfolio: dict) -> list:
    """
    Get personalized concepts for a student based on their assessment results.

    Args:
        student_id: Student ID
        topic_name: Topic name
        portfolio: Student portfolio data

    Returns:
        List of personalized concepts with priority
    """
    # Get topic progress
    topic_progress = portfolio.get('topic_progress', {}).get(topic_name, {})

    # Get all concepts for the topic
    if lab_tutor_loader:
        topic_summary = lab_tutor_loader.get_topic_summary(topic_name)
        all_concepts = topic_summary.get('concepts', [])
    else:
        all_concepts = []

    # Categorize concepts based on mastery
    personalized = []

    for concept in all_concepts:
        # Get mastery level from assessment results (if available)
        mastery = topic_progress.get('concept_mastery', {}).get(concept, 0.5)

        if mastery < 0.5:
            priority = 'high'
            status = 'needs_focus'
        elif mastery < 0.7:
            priority = 'medium'
            status = 'developing'
        else:
            priority = 'low'
            status = 'proficient'

        personalized.append({
            'name': concept,
            'mastery': mastery,
            'priority': priority,
            'status': status
        })

    # Sort by priority (high first)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    personalized.sort(key=lambda x: (priority_order[x['priority']], -x['mastery']))

    return personalized


def _create_personalized_learning_path(student_id: str, topic_name: str, concept_scores: dict) -> list:
    """
    Create a personalized learning path based on assessment results.

    Args:
        student_id: Student ID
        topic_name: Topic name
        concept_scores: Dictionary of concept scores from assessment

    Returns:
        List of personalized concepts with recommendations
    """
    personalized = []

    for concept, score in concept_scores.items():
        mastery = score  # Score is already a mastery estimate (0.0 to 1.0)

        # Determine priority and recommendations
        if mastery < 0.5:
            priority = 'high'
            status = 'needs_focus'
            recommendation = 'Start with foundational materials and examples'
        elif mastery < 0.7:
            priority = 'medium'
            status = 'developing'
            recommendation = 'Review materials and practice with exercises'
        else:
            priority = 'low'
            status = 'proficient'
            recommendation = 'Quick review and advanced challenges'

        personalized.append({
            'name': concept,
            'mastery': mastery,
            'priority': priority,
            'status': status,
            'recommendation': recommendation
        })

    # Sort by priority (high first)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    personalized.sort(key=lambda x: (priority_order[x['priority']], -x['mastery']))

    # Store in graph if available
    if graph_manager and graph_manager.neo4j:
        try:
            for concept_data in personalized:
                query = """
                MATCH (s:STUDENT {student_id: $student_id})
                MATCH (t:TOPIC {name: $topic_name})
                MERGE (c:CONCEPT {name: $concept_name})
                MERGE (c)-[:BELONGS_TO]->(t)
                MERGE (s)-[p:PERSONALIZED_PATH]->(c)
                SET p.mastery = $mastery,
                    p.priority = $priority,
                    p.status = $status,
                    p.recommendation = $recommendation,
                    p.created_at = datetime()
                """
                graph_manager.neo4j.execute_query(query, {
                    'student_id': student_id,
                    'topic_name': topic_name,
                    'concept_name': concept_data['name'],
                    'mastery': concept_data['mastery'],
                    'priority': concept_data['priority'],
                    'status': concept_data['status'],
                    'recommendation': concept_data['recommendation']
                })
        except Exception as e:
            print(f"Error storing personalized path: {e}")

    return personalized


# ==================== CHATBOT API ====================

@student_learning_bp.route('/api/chat', methods=['POST'])
def chatbot_api(student_id):
    """
    API endpoint for BIT Tutor AI chatbot - available globally on all pages
    """
    data = request.get_json()
    message = data.get('message', '')
    topic = data.get('topic', '')
    context = data.get('context', 'general')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Use learning assistant if available
        if learning_assistant:
            response = learning_assistant.get_response(
                student_id=student_id,
                message=message,
                topic=topic,
                context=context
            )
        else:
            # Fallback response
            response = f"I understand you're asking about '{message}'. Let me help you with that. "
            if topic:
                response += f"Since you're learning about {topic}, I recommend reviewing the reading materials and videos for this topic."
            else:
                response += "Please let me know which topic you'd like to learn more about!"

        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in chatbot API: {e}")
        return jsonify({
            'success': False,
            'response': 'Sorry, I encountered an error. Please try again.',
            'error': str(e)
        }), 500


# ==================== GRADE TRACKING ====================

@student_learning_bp.route('/api/grades/record', methods=['POST'])
def record_grade(student_id):
    """
    Record student grade in Neo4j knowledge graph
    """
    data = request.get_json()
    assessment_type = data.get('type', 'quiz')  # quiz, lab, assessment
    topic_name = data.get('topic')
    score = data.get('score', 0)
    max_score = data.get('max_score', 100)
    details = data.get('details', {})

    if not graph_manager or not graph_manager.neo4j:
        return jsonify({'error': 'Graph manager not available'}), 500

    try:
        # Create grade node and link to student
        grade_query = """
        MATCH (s:Student {student_id: $student_id})
        OPTIONAL MATCH (t:Topic {name: $topic_name})
        CREATE (g:Grade {
            grade_id: $grade_id,
            type: $type,
            score: $score,
            max_score: $max_score,
            percentage: $percentage,
            date: datetime(),
            details: $details
        })
        CREATE (s)-[:EARNED]->(g)
        WITH g, t
        WHERE t IS NOT NULL
        CREATE (g)-[:FOR_TOPIC]->(t)
        RETURN g.grade_id as grade_id
        """

        grade_id = f"grade_{student_id}_{assessment_type}_{datetime.now().timestamp()}"
        percentage = (score / max_score * 100) if max_score > 0 else 0

        result = graph_manager.neo4j.graph.query(grade_query, {
            'student_id': student_id,
            'topic_name': topic_name,
            'grade_id': grade_id,
            'type': assessment_type,
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'details': str(details)
        })

        # Update student's overall score
        update_student_query = """
        MATCH (s:Student {student_id: $student_id})-[:EARNED]->(g:Grade)
        WITH s, avg(g.percentage) as avg_score
        SET s.overall_score = avg_score
        RETURN s.overall_score as overall_score
        """

        graph_manager.neo4j.graph.query(update_student_query, {'student_id': student_id})

        return jsonify({
            'success': True,
            'grade_id': grade_id,
            'percentage': percentage,
            'message': 'Grade recorded successfully'
        })

    except Exception as e:
        print(f"Error recording grade: {e}")
        return jsonify({'error': str(e)}), 500


@student_learning_bp.route('/api/grades', methods=['GET'])
def get_grades(student_id):
    """
    Get all grades for a student from student attributes
    """
    if not graph_manager or not graph_manager.neo4j:
        return jsonify({'error': 'Graph manager not available'}), 500

    try:
        # Get grades from student attributes
        grades_query = """
        MATCH (s:Student {student_id: $student_id})
        RETURN s.grades as grades_json,
               s.total_grades as total_grades,
               s.overall_score as overall_score
        """
        result = graph_manager.neo4j.graph.query(grades_query, {'student_id': student_id})

        grades = []
        final_score = 0

        if result and len(result) > 0:
            record = result[0]
            grades_json = record.get('grades_json')

            if grades_json:
                import json
                try:
                    grades = json.loads(grades_json)
                except:
                    grades = []

            # Calculate final score (average of all grades)
            if grades:
                final_score = sum(g.get('percentage', 0) for g in grades) / len(grades)
            else:
                final_score = record.get('overall_score', 0)

        return jsonify({
            'success': True,
            'grades': grades,
            'total_grades': len(grades),
            'final_score': round(final_score, 1)
        })

    except Exception as e:
        print(f"Error fetching grades: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'grades': [], 'total_grades': 0, 'final_score': 0}), 500


@student_learning_bp.route('/topics/<topic_name>/concept/<concept_name>/blog')
def concept_reading_blog(student_id, topic_name, concept_name):
    """
    Generate and display a personalized blog for a concept using student hobbies.
    """
    if not graph_manager:
        return "Graph manager not available", 500

    try:
        # Get student hobbies and interests
        student_query = """
        MATCH (s:Student {student_id: $student_id})
        RETURN s.name as name, s.hobbies as hobbies, s.interests as interests
        """
        student_result = graph_manager.neo4j.graph.query(student_query, {'student_id': student_id})

        student_name = "Student"
        student_hobbies = []
        student_interests = []

        if student_result and len(student_result) > 0:
            student_name = student_result[0].get('name', 'Student')
            student_hobbies = student_result[0].get('hobbies', [])
            student_interests = student_result[0].get('interests', [])

        # Get concept and theory data
        concept_query = """
        MATCH (c:Concept {name: $concept_name})
        OPTIONAL MATCH (th:Theory)-[:CONSISTS_OF]->(c)
        OPTIONAL MATCH (c)-[:EXPLAINED_BY]->(r:ReadingMaterial)
        RETURN c.name as concept_name,
               c.description as description,
               th.compressed_text as theory_text,
               th.keywords as keywords,
               r.title as reading_title,
               r.source as reading_source
        LIMIT 1
        """
        concept_result = graph_manager.neo4j.graph.query(
            concept_query,
            {'concept_name': concept_name}
        )

        if not concept_result or len(concept_result) == 0:
            return "Concept not found", 404

        concept_data = dict(concept_result[0])

        # Generate personalized blog content
        blog_content = _generate_personalized_blog(
            concept_name=concept_name,
            concept_data=concept_data,
            student_hobbies=student_hobbies,
            student_interests=student_interests,
            topic_name=topic_name
        )

        return render_template('student/concept_blog.html',
                             student_id=student_id,
                             student_name=student_name,
                             topic_name=topic_name,
                             concept_name=concept_name,
                             blog_content=blog_content,
                             hobbies=student_hobbies,
                             interests=student_interests)

    except Exception as e:
        print(f"Error generating blog: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating blog: {str(e)}", 500


def _generate_personalized_blog(concept_name, concept_data, student_hobbies, student_interests, topic_name):
    """
    Generate personalized blog content using AI and student hobbies/interests.
    Uses LLM to create original, engaging educational content.
    """
    import random
    import os

    theory_text = concept_data.get('theory_text', '')
    description = concept_data.get('description', '')

    # Select primary hobby and interest for personalization
    primary_hobby = random.choice(student_hobbies) if student_hobbies else "everyday life"
    primary_interest = random.choice(student_interests) if student_interests else "technology"

    # Try to use LLM for generation
    try:
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')

        print(f"🤖 Blog Generation Debug:")
        print(f"   - OpenAI API Key present: {bool(api_key)}")
        print(f"   - Concept: {concept_name}")
        print(f"   - Hobby: {primary_hobby}")
        print(f"   - Interest: {primary_interest}")

        if api_key:
            print(f"   ✅ Using AI generation (GPT-4)")
            client = OpenAI(api_key=api_key)

            # Create personalized prompt
            prompt = f"""Write an engaging, educational blog post about "{concept_name}" in the context of "{topic_name}".

PERSONALIZATION:
- Student's hobby: {primary_hobby}
- Student's interest: {primary_interest}
- Additional hobbies: {', '.join(student_hobbies[:3]) if len(student_hobbies) > 1 else 'none'}

CONCEPT INFORMATION:
{theory_text or description or 'No additional context provided'}

REQUIREMENTS:
1. Write in a conversational, engaging tone
2. Start with a compelling hook that relates to the student's hobby ({primary_hobby})
3. Explain the concept clearly with real-world examples from {primary_hobby} and {primary_interest}
4. Include practical applications and use cases
5. Add 3-4 concrete examples that connect to the student's interests
6. Provide key takeaways (3-5 bullet points)
7. Suggest next steps for learning
8. Keep it educational but fun and relatable

LENGTH: 600-800 words

FORMAT:
Write as a cohesive blog post with natural paragraphs. Don't use section headers like "INTRODUCTION:" or "THEORY:".
Just write engaging content that flows naturally.

TONE: Friendly, educational, and personalized to someone interested in {primary_hobby} and {primary_interest}."""

            # Generate with GPT-4
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator who writes engaging, personalized learning materials. You excel at making complex technical concepts accessible through relatable examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for more creative content
                max_tokens=1500
            )

            generated_content = response.choices[0].message.content.strip()

            # Parse the generated content into structured format
            blog = {
                'title': f"Understanding {concept_name}",
                'subtitle': f"A personalized guide for {topic_name}",
                'introduction': f"Customized for your interests in {primary_hobby} and {primary_interest}",
                'content': generated_content,  # Full AI-generated content
                'theory': theory_text or description,
                'personalization': {
                    'hobby': primary_hobby,
                    'interest': primary_interest,
                    'all_hobbies': student_hobbies,
                    'all_interests': student_interests
                },
                'generated_by': 'AI (GPT-4)',
                'next_steps': [
                    f"Practice applying {concept_name} in your {primary_hobby} projects",
                    f"Explore how {concept_name} is used in {primary_interest}",
                    f"Try creating your own examples related to {primary_hobby}"
                ]
            }

            print(f"   ✅ AI blog generated successfully!")
            return blog
        else:
            print(f"   ⚠️  No OpenAI API key found. Using template generation.")

    except ImportError as e:
        print(f"⚠️  OpenAI package not installed: {e}. Using template-based generation.")
    except Exception as e:
        print(f"⚠️  Error generating with AI: {e}. Falling back to template.")
        import traceback
        traceback.print_exc()

    # Fallback: Enhanced template-based generation
    print(f"   📝 Using template-based generation")
    return _generate_template_blog(concept_name, concept_data, student_hobbies, student_interests, topic_name)


def _generate_template_blog(concept_name, concept_data, student_hobbies, student_interests, topic_name):
    """
    Fallback template-based blog generation with better structure.
    """
    import random

    theory_text = concept_data.get('theory_text', '')
    description = concept_data.get('description', '')

    hobby = random.choice(student_hobbies) if student_hobbies else "everyday life"
    interest = random.choice(student_interests) if student_interests else "technology"

    # Generate personalized examples
    examples = _generate_hobby_based_examples(concept_name, hobby, interest)

    # Create enhanced blog structure
    blog = {
        'title': f"Understanding {concept_name}",
        'subtitle': f"A personalized guide for {topic_name}",
        'introduction': f"Let's explore {concept_name} through examples from {hobby} and {interest}!",
        'theory': theory_text or description or f"{concept_name} is a fundamental concept in {topic_name}.",
        'examples': examples,
        'summary': f"In summary, {concept_name} is essential for understanding {topic_name}. By relating it to {hobby}, we can see how it applies in real-world scenarios.",
        'personalization': {
            'hobby': hobby,
            'interest': interest,
            'all_hobbies': student_hobbies,
            'all_interests': student_interests
        },
        'generated_by': 'Template',
        'next_steps': [
            f"Practice applying {concept_name} in different contexts",
            f"Explore how {concept_name} relates to other concepts in {topic_name}",
            f"Try creating your own examples using {hobby}"
        ]
    }

    return blog


def _generate_hobby_based_examples(concept_name, hobby, interest):
    """
    Generate examples based on student hobbies and interests.
    """
    # Enhanced hobby-based example templates
    hobby_examples = {
        'sports': [
            f"In {hobby}, {concept_name} is like organizing a team roster - you need to structure player data efficiently.",
            f"Think of {concept_name} as a playbook in {hobby} - it defines how different plays (data) are organized and executed."
        ],
        'music': [
            f"In {hobby}, {concept_name} is similar to organizing a music library - different genres, artists, and albums need flexible organization.",
            f"Consider {concept_name} like a playlist in {hobby} - you can organize songs in various ways without a rigid structure."
        ],
        'gaming': [
            f"In {hobby}, {concept_name} is like a game's inventory system - items need to be stored and retrieved quickly.",
            f"Think of {concept_name} as a save file in {hobby} - it stores game state in a structured but flexible way."
        ],
        'cooking': [
            f"In {hobby}, {concept_name} is like organizing recipes - ingredients and steps can vary greatly between dishes.",
            f"Consider {concept_name} as a pantry in {hobby} - different items are stored based on their type and usage."
        ],
        'travel': [
            f"In {hobby}, {concept_name} is like planning an itinerary - destinations and activities need flexible organization.",
            f"Think of {concept_name} as a travel journal in {hobby} - experiences are recorded in various formats."
        ],
        'reading': [
            f"In {hobby}, {concept_name} is like organizing a personal library - books can be categorized in multiple ways.",
            f"Consider {concept_name} as a reading list in {hobby} - books are organized by genre, author, or reading order."
        ],
        'art': [
            f"In {hobby}, {concept_name} is like organizing an art portfolio - different pieces need flexible categorization.",
            f"Think of {concept_name} as a color palette in {hobby} - colors are organized for quick access and combination."
        ],
        'technology': [
            f"In {hobby}, {concept_name} is like organizing software projects - code and data need efficient structure.",
            f"Consider {concept_name} as a file system in {hobby} - files are organized hierarchically for easy access."
        ]
    }

    # Get examples for the hobby, or use generic examples
    examples = hobby_examples.get(hobby.lower(), [
        f"In {hobby}, {concept_name} helps organize and manage information efficiently.",
        f"Think of {concept_name} as a way to structure data related to {hobby}."
    ])

    # Add interest-based context
    examples.append(f"This concept is particularly relevant in {interest}, where efficient data management is crucial.")

    return examples

