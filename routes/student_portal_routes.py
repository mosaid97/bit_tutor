"""
Student Portal Routes
Handles student authentication, registration, and portal access
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from datetime import datetime
from services.analytics.performance_analyzer import get_performance_analyzer

# Create blueprint
student_portal_bp = Blueprint('student_portal', __name__, url_prefix='/student')

# Global references (will be set by main app)
graph_manager = None
auth_service = None


def init_portal_routes(gm, auth_svc):
    """Initialize routes with graph manager and auth service"""
    global graph_manager, auth_service
    graph_manager = gm
    auth_service = auth_svc


@student_portal_bp.route('/')
def portal_home():
    """Student portal landing page - Sign In or Register"""
    # Check if already logged in
    if 'student_id' in session:
        return redirect(url_for('student_portal.dashboard', student_id=session['student_id']))
    
    return render_template('student/portal_home.html')


@student_portal_bp.route('/register', methods=['GET'])
def register_page():
    """Display registration form"""
    # Get available classes
    classes = []
    if auth_service:
        classes = auth_service.get_available_classes()

    return render_template('student/register.html', classes=classes)


@student_portal_bp.route('/register', methods=['POST'])
def register_submit():
    """Process student registration"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['name', 'email', 'password', 'class_id']
        for field in required:
            if field not in data or not data[field]:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Register student
        result = auth_service.register_student(
            name=data['name'],
            email=data['email'],
            password=data['password'],
            class_id=data['class_id'],
            hobbies=data.get('hobbies', []),
            interests=data.get('interests', [])
        )
        
        if result['success']:
            # Set session
            session['student_id'] = result['student_id']
            session['student_name'] = data['name']
            session['student_email'] = data['email']
            
            return jsonify({
                'success': True,
                'message': 'Registration successful',
                'redirect': url_for('student_portal.dashboard', student_id=result['student_id'])
            })
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Registration error: {str(e)}'
        }), 500


@student_portal_bp.route('/login', methods=['GET'])
def login_page():
    """Display login form"""
    return render_template('student/login.html')


@student_portal_bp.route('/login', methods=['POST'])
def login_submit():
    """Process student login"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'email' not in data or 'password' not in data:
            return jsonify({
                'success': False,
                'message': 'Email and password required'
            }), 400
        
        # Authenticate student
        result = auth_service.login_student(
            email=data['email'],
            password=data['password']
        )
        
        if result['success']:
            # Set session
            session['student_id'] = result['student_id']
            session['student_name'] = result['name']
            session['student_email'] = result['email']
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'redirect': url_for('student_portal.dashboard', student_id=result['student_id'])
            })
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Login error: {str(e)}'
        }), 500


@student_portal_bp.route('/logout')
def logout():
    """Logout student"""
    session.clear()
    return redirect(url_for('student_portal.portal_home'))


@student_portal_bp.route('/<student_id>/select-class')
def select_class(student_id):
    """Class selection page - students choose which class to view"""
    # Check if logged in
    if 'student_id' not in session or session['student_id'] != student_id:
        return redirect(url_for('student_portal.login_page'))

    # Get student info
    student_info = auth_service.get_student_info(student_id)

    if not student_info:
        return redirect(url_for('student_portal.login_page'))

    # Get classes the student is enrolled in
    classes = []
    if graph_manager and graph_manager.neo4j:
        try:
            query = """
            MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)
            OPTIONAL MATCH (c)-[:INCLUDES]->(t:Topic)
            WITH c, count(DISTINCT t) as topic_count
            RETURN c.class_id as class_id,
                   c.name as name,
                   c.description as description,
                   c.semester as semester,
                   c.year as year,
                   topic_count
            ORDER BY c.year DESC, c.semester DESC
            """

            result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})

            if result:
                for record in result:
                    classes.append(dict(record))

        except Exception as e:
            print(f"Error fetching classes: {e}")

    return render_template('student/select_class.html',
                         student=student_info,
                         classes=classes)


@student_portal_bp.route('/<student_id>/dashboard')
def dashboard(student_id):
    """Student dashboard - main learning portal"""
    # Check if logged in
    if 'student_id' not in session or session['student_id'] != student_id:
        return redirect(url_for('student_portal.login_page'))

    # Get class_id from query parameter or session
    class_id = request.args.get('class_id')
    if class_id:
        session['selected_class_id'] = class_id
    elif 'selected_class_id' not in session:
        # No class selected, redirect to class selection
        return redirect(url_for('student_portal.select_class', student_id=student_id))
    else:
        class_id = session['selected_class_id']

    # Get student info
    student_info = auth_service.get_student_info(student_id)

    if not student_info:
        return redirect(url_for('student_portal.login_page'))

    # Check if student has completed class pre-assessment
    if graph_manager and graph_manager.neo4j:
        try:
            check_query = """
            MATCH (s:Student {student_id: $student_id})
            RETURN s.class_assessment_completed as completed
            """
            check_result = graph_manager.neo4j.graph.query(check_query, {'student_id': student_id})

            if not check_result or not check_result[0].get('completed'):
                # Redirect to class pre-assessment
                return redirect(f'/student/{student_id}/learn/class-assessment')
        except Exception as e:
            print(f"Error checking assessment status: {e}")

    # Get class info
    class_info = None
    if graph_manager and graph_manager.neo4j:
        try:
            class_query = """
            MATCH (c:Class {class_id: $class_id})
            RETURN c.class_id as class_id,
                   c.name as name,
                   c.description as description
            """
            result = graph_manager.neo4j.graph.query(class_query, {'class_id': class_id})
            if result and len(result) > 0:
                class_info = dict(result[0])
        except Exception as e:
            print(f"Error fetching class info: {e}")

    # Get topics from knowledge graph for the selected class
    topics = []
    stats = {
        'total_topics': 0,
        'total_concepts': 0,
        'total_quizzes': 0,
        'total_labs': 0
    }

    if graph_manager and graph_manager.neo4j:
        try:
            query = """
            MATCH (c:Class {class_id: $class_id})-[:INCLUDES]->(t:Topic)
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

            result = graph_manager.neo4j.graph.query(query, {'class_id': class_id})

            if result:
                for record in result:
                    topics.append(dict(record))
                    stats['total_concepts'] += record['concept_count']
                    stats['total_quizzes'] += record['quiz_count']
                    stats['total_labs'] += record['lab_count']

                stats['total_topics'] = len(topics)

        except Exception as e:
            print(f"Error fetching topics: {e}")

    return render_template('student/dashboard.html',
                         student=student_info,
                         class_info=class_info,
                         topics=topics,
                         stats=stats)


@student_portal_bp.route('/api/<student_id>/topics')
def get_topics_api(student_id):
    """API endpoint to get student topics"""
    # Check if logged in
    if 'student_id' not in session or session['student_id'] != student_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    topics = []
    total_concepts = 0
    total_quizzes = 0
    total_labs = 0
    
    if graph_manager and graph_manager.neo4j:
        try:
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
                    topic_dict = dict(record)
                    topics.append(topic_dict)
                    total_concepts += record['concept_count']
                    total_quizzes += record['quiz_count']
                    total_labs += record['lab_count']
                    
        except Exception as e:
            print(f"Error fetching topics: {e}")
    
    return jsonify({
        'topics': topics,
        'total_topics': len(topics),
        'total_concepts': total_concepts,
        'total_quizzes': total_quizzes,
        'total_labs': total_labs,
        'student_id': student_id
    })


@student_portal_bp.route('/api/<student_id>/analytics')
def get_student_analytics(student_id):
    """API endpoint for student analytics - for nexus dashboard"""
    # Check authentication
    if 'student_id' not in session or session['student_id'] != student_id:
        return jsonify({'error': 'Unauthorized'}), 401

    # Generate analytics data with actual student data
    analytics = {
        'performance_trend': {
            'dates': [],
            'scores': []
        },
        'skill_mastery': {},
        'learning_velocity': {
            'concepts_per_week': 0,
            'practice_hours': 0,
            'completion_rate': 0,
            'retention_score': 0
        },
        'engagement_patterns': {
            'hours': list(range(8, 24)),
            'activity_levels': [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3]
        },
        'ai_insights': [],
        'next_recommendations': []
    }

    # Get skill mastery from Neo4j - use actual student topic mastery
    if graph_manager and graph_manager.neo4j:
        try:
            # Get topic-level mastery for spider web (based on student's KNOWS relationships)
            topic_mastery_query = """
            MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)-[:INCLUDES]->(t:Topic)
            OPTIONAL MATCH (s)-[k:KNOWS]->(con:Concept)<-[:INCLUDES_CONCEPT]-(t)
            WITH t.name as topic_name,
                 AVG(COALESCE(k.mastery_level, 0.0)) * 100 as avg_mastery
            RETURN topic_name, avg_mastery
            ORDER BY topic_name
            LIMIT 8
            """
            result = graph_manager.neo4j.graph.query(topic_mastery_query, {'student_id': student_id})

            if result and len(result) > 0:
                for record in result:
                    analytics['skill_mastery'][record['topic_name']] = round(record['avg_mastery'], 1)
            else:
                # If no mastery data, show topics with 0% mastery
                topics_query = """
                MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)-[:INCLUDES]->(t:Topic)
                RETURN t.name as topic_name
                LIMIT 8
                """
                topics_result = graph_manager.neo4j.graph.query(topics_query, {'student_id': student_id})
                if topics_result:
                    for topic in topics_result:
                        analytics['skill_mastery'][topic['topic_name']] = 0
                else:
                    # Absolute fallback
                    analytics['skill_mastery'] = {
                        'Complete Pre-Assessment': 0
                    }

            # Get performance trend by day (last 14 days)
            performance_query = """
            MATCH (g:StudentScore {student_id: $student_id})
            WHERE g.completed_at >= datetime() - duration('P14D')
            WITH date(g.completed_at) as score_date, AVG(g.percentage) as avg_score
            RETURN score_date, avg_score
            ORDER BY score_date
            """

            perf_result = graph_manager.neo4j.graph.query(performance_query, {'student_id': student_id})

            if perf_result and len(perf_result) > 0:
                raw_dates = [record['score_date'].strftime('%b %d') for record in perf_result]
                raw_scores = [round(record['avg_score'], 1) for record in perf_result]

                # Use Performance Analyzer for advanced trend analysis
                analyzer = get_performance_analyzer(raw_dates, raw_scores)
                trend_analysis = analyzer.analyze()

                # Update analytics with analyzed data
                analytics['performance_trend'] = {
                    'dates': trend_analysis['dates'],
                    'scores': trend_analysis['smoothed_scores'],  # EMA smoothed
                    'raw_scores': trend_analysis['raw_scores'],
                    'trend_line': trend_analysis['trend_line'],  # Linear regression
                    'predictions': trend_analysis['predictions'],  # Next 3 days
                    'trend_direction': trend_analysis['trend_direction'],
                    'slope': trend_analysis['slope'],
                    'anomalies': trend_analysis['anomalies']
                }

                # Add trend insights to AI insights
                if trend_analysis['insights']:
                    analytics['ai_insights'].extend(trend_analysis['insights'])
            else:
                # Fallback: use last 7 days with placeholder data
                from datetime import timedelta
                today = datetime.now()
                analytics['performance_trend']['dates'] = [(today - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
                analytics['performance_trend']['scores'] = [0] * 7
                analytics['performance_trend']['raw_scores'] = [0] * 7
                analytics['performance_trend']['trend_line'] = [0] * 7
                analytics['performance_trend']['predictions'] = [0, 0, 0]
                analytics['performance_trend']['trend_direction'] = 'unknown'
                analytics['performance_trend']['slope'] = 0
                analytics['performance_trend']['anomalies'] = []

            # Calculate learning velocity metrics
            velocity_query = """
            MATCH (s:Student {student_id: $student_id})
            OPTIONAL MATCH (s)-[k:KNOWS]->(c:Concept)
            WHERE k.last_assessed >= datetime() - duration('P7D')
            WITH s, count(DISTINCT c) as concepts_this_week
            OPTIONAL MATCH (s)-[:REGISTERED_IN]->(cl:Class)-[:INCLUDES]->(t:Topic)
            WITH s, concepts_this_week,
                 count(DISTINCT t) as total_topics,
                 sum(CASE WHEN t.completed THEN 1 ELSE 0 END) as completed_topics
            RETURN concepts_this_week,
                   COALESCE(s.total_practice_hours, 0) as practice_hours,
                   CASE WHEN total_topics > 0
                        THEN (toFloat(completed_topics) * 100.0 / toFloat(total_topics))
                        ELSE 0 END as completion_rate,
                   COALESCE(s.overall_score, 0) as retention_score
            """

            velocity_result = graph_manager.neo4j.graph.query(velocity_query, {'student_id': student_id})

            if velocity_result and len(velocity_result) > 0:
                velocity_data = velocity_result[0]
                analytics['learning_velocity'] = {
                    'concepts_per_week': velocity_data['concepts_this_week'],
                    'practice_hours': round(velocity_data['practice_hours'], 1),
                    'completion_rate': round(velocity_data['completion_rate'], 1),
                    'retention_score': round(velocity_data['retention_score'], 1)
                }

                # Generate AI insights based on actual data
                insights = []

                # Insight about overall performance
                if velocity_data['retention_score'] >= 80:
                    insights.append("🎯 Excellent work! Your retention score is outstanding!")
                elif velocity_data['retention_score'] >= 60:
                    insights.append("📈 Good progress! Keep practicing to improve your retention.")
                else:
                    insights.append("💡 Focus on reviewing concepts to improve retention.")

                # Insight about learning velocity
                if velocity_data['concepts_this_week'] >= 5:
                    insights.append(f"🔥 You're on fire! {velocity_data['concepts_this_week']} concepts mastered this week!")
                elif velocity_data['concepts_this_week'] > 0:
                    insights.append(f"📚 {velocity_data['concepts_this_week']} concepts learned this week. Keep it up!")

                # Insight about completion rate
                if velocity_data['completion_rate'] >= 75:
                    insights.append("✨ You're making excellent progress through the course!")
                elif velocity_data['completion_rate'] >= 50:
                    insights.append("🎓 Halfway there! Keep pushing forward!")

                analytics['ai_insights'] = insights if insights else ["Start learning to get personalized insights!"]

            # Generate recommendations based on weak areas
            recommendations = []
            weak_topics_query = """
            MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept)<-[:INCLUDES_CONCEPT]-(t:Topic)
            WHERE k.mastery_level < 0.6
            WITH t.name as topic_name, AVG(k.mastery_level) * 100 as avg_mastery, count(c) as weak_concepts
            RETURN topic_name, avg_mastery, weak_concepts
            ORDER BY avg_mastery ASC
            LIMIT 3
            """

            weak_topics_result = graph_manager.neo4j.graph.query(weak_topics_query, {'student_id': student_id})
            if weak_topics_result and len(weak_topics_result) > 0:
                for record in weak_topics_result:
                    recommendations.append({
                        'title': f"Review {record['topic_name']}",
                        'description': f"Focus on mastering {record['weak_concepts']} concepts",
                        'difficulty': 'Medium' if record['avg_mastery'] >= 40 else 'High Priority',
                        'xp_reward': 150,
                        'questions_count': record['weak_concepts']
                    })

            analytics['next_recommendations'] = recommendations if recommendations else [
                {
                    'title': 'Continue Learning',
                    'description': 'Explore new topics to expand your knowledge',
                    'difficulty': 'Beginner',
                    'xp_reward': 100,
                    'questions_count': 10
                }
            ]

        except Exception as e:
            print(f"Error fetching analytics data: {e}")
            import traceback
            traceback.print_exc()
            analytics['skill_mastery'] = {
                'Complete Pre-Assessment': 0
            }

    return jsonify(analytics)


@student_portal_bp.route('/<student_id>/assessment/initial')
def initial_assessment(student_id):
    """Initial cognitive assessment page"""
    # Check authentication
    if 'student_id' not in session or session['student_id'] != student_id:
        return redirect(url_for('student_portal.login_page'))

    return render_template('student/initial_assessment.html', student_id=student_id)


@student_portal_bp.route('/<student_id>/assessment/topic/<topic_name>')
def topic_assessment(student_id, topic_name):
    """Pre-topic assessment page"""
    # Check authentication
    if 'student_id' not in session or session['student_id'] != student_id:
        return redirect(url_for('student_portal.login_page'))

    return render_template('student/topic_assessment.html',
                         student_id=student_id,
                         topic_name=topic_name)


@student_portal_bp.route('/<student_id>/browse-topics')
def browse_topics(student_id):
    """Browse all available topics"""
    # Check authentication
    if 'student_id' not in session or session['student_id'] != student_id:
        return redirect(url_for('student_portal.login_page'))

    # Redirect to learning topics page
    return redirect(url_for('student_learning.browse_topics', student_id=student_id))


@student_portal_bp.route('/<student_id>/ai-tutor')
def ai_tutor(student_id):
    """AI Tutor page - dedicated chatbot interface"""
    # Check authentication
    if 'student_id' not in session or session['student_id'] != student_id:
        return redirect(url_for('student_portal.login_page'))

    return render_template('student/ai_tutor.html', student_id=student_id)


@student_portal_bp.route('/<student_id>/progress')
def cognitive_progress(student_id):
    """Cognitive progress dashboard with G-CDM visualization"""
    import time
    start_time = time.time()
    print(f"🔍 Progress route accessed for student: {student_id}")
    print(f"🔍 Session student_id: {session.get('student_id')}")

    # Check authentication
    if 'student_id' not in session or session['student_id'] != student_id:
        print(f"⚠️  Authentication failed - redirecting to login")
        return redirect(url_for('student_portal.login_page'))

    print(f"⏱️  Auth check: {time.time() - start_time:.2f}s")

    # Initialize data
    progress_data = {
        'overall_score': 0,
        'mastered_concepts': 0,
        'total_concepts': 0,
        'completed_topics': 0,
        'total_topics': 0,
        'streak_days': 0,
        'cognitive_profile': [],
        'topic_progress': [],
        'assessment_history': []
    }

    # Get student info
    student_info = auth_service.get_student_info(student_id) if auth_service else None

    if graph_manager and graph_manager.neo4j:
        try:
            # Get overall stats - OPTIMIZED QUERY
            stats_query = """
            MATCH (s:Student {student_id: $student_id})
            OPTIONAL MATCH (s)-[k:KNOWS]->(c:Concept)
            WITH s, count(DISTINCT k) as mastered_concepts
            OPTIONAL MATCH (s)-[:REGISTERED_IN]->(cl:Class)-[:INCLUDES]->(t:Topic)
            WITH s, mastered_concepts, count(DISTINCT t) as total_topics
            RETURN
                COALESCE(s.overall_score, 0) as overall_score,
                COALESCE(s.streak_days, 0) as streak_days,
                mastered_concepts,
                mastered_concepts as total_concepts,
                0 as completed_topics,
                total_topics
            """
            stats_result = graph_manager.neo4j.graph.query(stats_query, {'student_id': student_id})
            if stats_result:
                stats = stats_result[0]
                progress_data.update({
                    'overall_score': int(stats.get('overall_score', 0) or 0),
                    'mastered_concepts': int(stats.get('mastered_concepts', 0) or 0),
                    'total_concepts': int(stats.get('total_concepts', 0) or 0),
                    'completed_topics': int(stats.get('completed_topics', 0) or 0),
                    'total_topics': int(stats.get('total_topics', 0) or 0),
                    'streak_days': int(stats.get('streak_days', 0) or 0)
                })

            # Get cognitive profile (G-CDM) - mastery levels for all concepts
            profile_query = """
            MATCH (s:Student {student_id: $student_id})-[k:KNOWS]->(c:Concept)
            MATCH (c)<-[:INCLUDES_CONCEPT]-(t:Topic)
            RETURN c.concept_id as concept_id,
                   c.name as concept_name,
                   t.name as topic,
                   COALESCE(k.mastery_level, 0.0) * 100 as mastery_level,
                   CASE
                       WHEN k.mastery_level >= 0.8 THEN 'Excellent! Keep reinforcing this knowledge.'
                       WHEN k.mastery_level >= 0.6 THEN 'Good progress. Practice more to master it.'
                       WHEN k.mastery_level >= 0.4 THEN 'Needs improvement. Review the materials.'
                       ELSE 'Focus area. Start with fundamentals.'
                   END as recommendation,
                   k.last_assessed as last_assessed
            ORDER BY k.mastery_level ASC
            """
            profile_results = graph_manager.neo4j.graph.query(profile_query, {'student_id': student_id})
            if profile_results:
                cognitive_profile = []
                for p in profile_results:
                    profile_dict = dict(p)
                    # Round mastery level to 1 decimal
                    profile_dict['mastery_level'] = round(profile_dict['mastery_level'], 1)
                    cognitive_profile.append(profile_dict)
                progress_data['cognitive_profile'] = cognitive_profile

            # If no cognitive profile exists yet, create placeholder message
            if not progress_data['cognitive_profile']:
                progress_data['cognitive_profile'] = [
                    {
                        'concept_name': 'No assessment data yet',
                        'topic': 'Complete the class pre-assessment',
                        'mastery_level': 0,
                        'recommendation': 'Take the class pre-assessment to establish your cognitive profile.'
                    }
                ]

            # Get topic progress
            topic_query = """
            MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(cl:Class)-[:INCLUDES]->(t:Topic)
            OPTIONAL MATCH (s)-[k:KNOWS]->(c:Concept)<-[:INCLUDES_CONCEPT]-(t)
            WITH t, count(DISTINCT k) as known_concepts
            OPTIONAL MATCH (t)-[:INCLUDES_CONCEPT]->(all_c:Concept)
            WITH t, known_concepts, count(DISTINCT all_c) as total_concepts
            RETURN t.name as topic_name,
                   COALESCE(t.description, '') as description,
                   CASE WHEN total_concepts > 0
                        THEN toInteger((toFloat(known_concepts) / toFloat(total_concepts)) * 100)
                        ELSE 0 END as progress,
                   0 as score,
                   0 as videos_watched,
                   0 as total_videos,
                   0 as readings_completed,
                   0 as total_readings,
                   false as lab_completed,
                   false as quiz_completed,
                   false as completed,
                   known_concepts > 0 as in_progress
            ORDER BY progress DESC
            """
            topic_results = graph_manager.neo4j.graph.query(topic_query, {'student_id': student_id})
            if topic_results:
                progress_data['topic_progress'] = [dict(t) for t in topic_results]

            # Get assessment history
            assessment_query = """
            MATCH (s:Student {student_id: $student_id})-[t:TOOK]->(a:Assessment)
            RETURN a.date as date,
                   a.type as type,
                   COALESCE(a.topic, 'General') as topic,
                   COALESCE(a.score, 0) as score
            ORDER BY a.date DESC
            LIMIT 10
            """
            assessment_results = graph_manager.neo4j.graph.query(assessment_query, {'student_id': student_id})
            if assessment_results:
                progress_data['assessment_history'] = [dict(a) for a in assessment_results]

        except Exception as e:
            print(f"Error fetching progress data: {e}")
            import traceback
            traceback.print_exc()

    # Generate personalized recommendations based on weak areas
    recommendations = []
    if progress_data['cognitive_profile'] and progress_data['cognitive_profile'][0]['concept_name'] != 'No assessment data yet':
        # Get concepts with low mastery (< 60%)
        weak_concepts = [c for c in progress_data['cognitive_profile'] if c['mastery_level'] < 60]

        for concept in weak_concepts[:5]:  # Top 5 weak areas
            recommendations.append({
                'title': f"Master {concept['concept_name']}",
                'topic': concept['topic'],
                'reason': f"Current mastery: {concept['mastery_level']}%",
                'action': 'Start Learning',
                'priority': 'high' if concept['mastery_level'] < 40 else 'medium'
            })

        # Add topics where student hasn't started
        if graph_manager and graph_manager.neo4j:
            try:
                unstarted_query = """
                MATCH (s:Student {student_id: $student_id})-[:REGISTERED_IN]->(c:Class)-[:INCLUDES]->(t:Topic)
                WHERE NOT EXISTS {
                    MATCH (s)-[:KNOWS]->(:Concept)<-[:INCLUDES_CONCEPT]-(t)
                }
                RETURN t.name as topic_name, t.description as description
                LIMIT 3
                """
                unstarted_result = graph_manager.neo4j.graph.query(unstarted_query, {'student_id': student_id})
                for topic in unstarted_result:
                    recommendations.append({
                        'title': f"Explore {topic['topic_name']}",
                        'topic': topic['topic_name'],
                        'reason': 'New topic to explore',
                        'action': 'Start Topic',
                        'priority': 'low'
                    })
            except Exception as e:
                print(f"Error fetching unstarted topics: {e}")

    progress_data['recommendations'] = recommendations

    print(f"⏱️  Total processing time: {time.time() - start_time:.2f}s")
    print(f"📊 Rendering template with {len(progress_data['cognitive_profile'])} concepts")

    return render_template('student/progress_nexus.html',
                         student_id=student_id,
                         student=student_info,
                         topics_completed=progress_data['completed_topics'],
                         learning_streak=progress_data['streak_days'],
                         **progress_data)

