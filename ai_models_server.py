"""
AI Models Server - Dedicated API for Knowledge Tracing, Cognitive Diagnosis, and Recommendation
Runs as a separate microservice in Docker container
"""

from flask import Flask, request, jsonify
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# Initialize AI Services
# ============================================================================

try:
    from services.knowledge_tracing.services.sqkt_service import SQKTIntegrationService
    sqkt_service = SQKTIntegrationService()
    logger.info("✅ SQKT Service initialized")
except Exception as e:
    logger.warning(f"⚠️ SQKT Service initialization failed: {e}")
    sqkt_service = None

try:
    from services.cognitive_diagnosis.services.ad4cd_service import AD4CDIntegrationService
    ad4cd_service = AD4CDIntegrationService()
    logger.info("✅ AD4CD Service initialized")
except Exception as e:
    logger.warning(f"⚠️ AD4CD Service initialization failed: {e}")
    ad4cd_service = None

try:
    from services.recommendation.services.recommendation_service import get_recommendation_service
    recommendation_service = get_recommendation_service()
    logger.info("✅ Recommendation Service initialized")
except Exception as e:
    logger.warning(f"⚠️ Recommendation Service initialization failed: {e}")
    recommendation_service = None

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    services_status = {
        'sqkt': sqkt_service is not None,
        'ad4cd': ad4cd_service is not None,
        'recommendation': recommendation_service is not None
    }
    
    all_healthy = all(services_status.values())
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'services': services_status
    }), 200 if all_healthy else 503

# ============================================================================
# Knowledge Tracing Endpoints (SQKT + MLFBK)
# ============================================================================

@app.route('/api/kt/predict', methods=['POST'])
def kt_predict():
    """Predict student performance on next question"""
    if not sqkt_service:
        return jsonify({'error': 'SQKT service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        question_id = data.get('question_id')
        
        prediction = sqkt_service.predict_next_performance(student_id, question_id)
        
        return jsonify({
            'student_id': student_id,
            'question_id': question_id,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"KT prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/kt/concept-predictions', methods=['POST'])
def kt_concept_predictions():
    """Get concept-level mastery predictions"""
    if not sqkt_service:
        return jsonify({'error': 'SQKT service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        concepts = data.get('concepts', [])
        
        predictions = sqkt_service.get_concept_predictions(student_id, concepts)
        
        return jsonify({
            'student_id': student_id,
            'predictions': predictions,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Concept predictions error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/kt/skill-gaps', methods=['POST'])
def kt_skill_gaps():
    """Identify skill gaps for targeted practice"""
    if not sqkt_service:
        return jsonify({'error': 'SQKT service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        concepts = data.get('concepts', [])
        target_mastery = data.get('target_mastery', 0.7)
        
        gaps = sqkt_service.get_skill_gaps(student_id, concepts, target_mastery)
        
        return jsonify({
            'student_id': student_id,
            'skill_gaps': gaps,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Skill gaps error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/kt/trajectory', methods=['POST'])
def kt_trajectory():
    """Get learning trajectory over time"""
    if not sqkt_service:
        return jsonify({'error': 'SQKT service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        
        trajectory = sqkt_service.get_learning_trajectory(student_id)
        
        return jsonify({
            'student_id': student_id,
            'trajectory': trajectory,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Trajectory error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Cognitive Diagnosis Endpoints (G-CDM + AD4CD)
# ============================================================================

@app.route('/api/cd/diagnose', methods=['POST'])
def cd_diagnose():
    """Diagnose student response with anomaly detection"""
    if not ad4cd_service:
        return jsonify({'error': 'AD4CD service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        question_id = data.get('question_id')
        response = data.get('response')
        
        diagnosis = ad4cd_service.diagnose_response(student_id, question_id, response)
        
        return jsonify({
            'student_id': student_id,
            'question_id': question_id,
            'diagnosis': diagnosis,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cd/mastery-profile', methods=['POST'])
def cd_mastery_profile():
    """Get detailed mastery profile for concepts"""
    if not ad4cd_service:
        return jsonify({'error': 'AD4CD service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        concepts = data.get('concepts', [])
        
        profile = ad4cd_service.get_mastery_profile(student_id, concepts)
        
        return jsonify({
            'student_id': student_id,
            'mastery_profile': profile,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Mastery profile error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cd/anomaly-report', methods=['POST'])
def cd_anomaly_report():
    """Get anomaly detection report"""
    if not ad4cd_service:
        return jsonify({'error': 'AD4CD service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        
        report = ad4cd_service.get_anomaly_report(student_id)
        
        return jsonify({
            'student_id': student_id,
            'anomaly_report': report,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Anomaly report error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cd/insights', methods=['POST'])
def cd_insights():
    """Get diagnostic insights (strengths, weaknesses, misconceptions)"""
    if not ad4cd_service:
        return jsonify({'error': 'AD4CD service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        concepts = data.get('concepts', [])
        
        insights = ad4cd_service.get_diagnostic_insights(student_id, concepts)
        
        return jsonify({
            'student_id': student_id,
            'insights': insights,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Insights error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Recommendation Endpoints (RL Agent)
# ============================================================================

@app.route('/api/rec/recommend', methods=['POST'])
def rec_recommend():
    """Get personalized content recommendation"""
    if not recommendation_service:
        return jsonify({'error': 'Recommendation service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        current_topic = data.get('current_topic')
        
        recommendation = recommendation_service.get_recommendation(student_id, current_topic)
        
        return jsonify({
            'student_id': student_id,
            'recommendation': recommendation,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rec/content-recommendations', methods=['POST'])
def rec_content():
    """Get ranked content recommendations"""
    if not recommendation_service:
        return jsonify({'error': 'Recommendation service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        concepts = data.get('concepts', [])
        
        recommendations = recommendation_service.get_content_recommendations(student_id, concepts)
        
        return jsonify({
            'student_id': student_id,
            'content_recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Content recommendations error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rec/hobby-personalization', methods=['POST'])
def rec_hobby():
    """Get hobby-based personalization"""
    if not recommendation_service:
        return jsonify({'error': 'Recommendation service not available'}), 503
    
    try:
        data = request.json
        student_id = data.get('student_id')
        concepts = data.get('concepts', [])
        
        personalization = recommendation_service.get_hobby_personalization(student_id, concepts)
        
        return jsonify({
            'student_id': student_id,
            'hobby_personalization': personalization,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Hobby personalization error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('AI_MODELS_PORT', 5000))
    logger.info(f"🚀 Starting AI Models Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

