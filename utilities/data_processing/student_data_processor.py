# utilities/data_processing/student_data_processor.py

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

class StudentDataProcessor:
    """
    Utility class for processing and managing student data across the BIT Tutor system.
    Handles data validation, transformation, and standardization.
    """
    
    def __init__(self):
        self.required_student_fields = [
            'student_id', 'name', 'email', 'current_focus', 'learning_style', 
            'difficulty_preference', 'hobbies', 'mastery_profile'
        ]
        print("StudentDataProcessor initialized.")
    
    def validate_student_data(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize student data structure.
        
        Args:
            student_data (dict): Raw student data
            
        Returns:
            dict: Validated and standardized student data
        """
        try:
            validated_data = {}
            
            # Validate required fields
            for field in self.required_student_fields:
                if field not in student_data:
                    if field == 'mastery_profile':
                        validated_data[field] = {}
                    elif field == 'hobbies':
                        validated_data[field] = []
                    elif field in ['current_focus', 'learning_style', 'difficulty_preference']:
                        validated_data[field] = 'medium'
                    else:
                        validated_data[field] = f"default_{field}"
                else:
                    validated_data[field] = student_data[field]
            
            # Validate data types
            if not isinstance(validated_data['hobbies'], list):
                validated_data['hobbies'] = [validated_data['hobbies']] if validated_data['hobbies'] else []
            
            if not isinstance(validated_data['mastery_profile'], dict):
                validated_data['mastery_profile'] = {}
            
            # Add metadata
            validated_data['last_updated'] = datetime.now().isoformat()
            validated_data['data_version'] = '1.0'
            
            return validated_data
            
        except Exception as e:
            print(f"Error validating student data: {e}")
            return self._get_default_student_data()
    
    def _get_default_student_data(self) -> Dict[str, Any]:
        """Get default student data structure."""
        return {
            'student_id': 'unknown',
            'name': 'Unknown Student',
            'email': 'unknown@example.com',
            'current_focus': 'python_basics',
            'learning_style': 'visual',
            'difficulty_preference': 'medium',
            'hobbies': [],
            'mastery_profile': {},
            'last_updated': datetime.now().isoformat(),
            'data_version': '1.0'
        }
    
    def process_interaction_data(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate interaction data.
        
        Args:
            interaction_data (dict): Raw interaction data
            
        Returns:
            dict: Processed interaction data
        """
        try:
            processed_data = {
                'interaction_id': interaction_data.get('interaction_id', f"int_{datetime.now().timestamp()}"),
                'student_id': interaction_data.get('student_id', 'unknown'),
                'type': interaction_data.get('type', 'unknown'),
                'timestamp': interaction_data.get('timestamp', datetime.now().isoformat()),
                'metadata': interaction_data.get('metadata', {})
            }
            
            # Process specific interaction types
            if processed_data['type'] == 'code_submission':
                processed_data.update({
                    'exercise_id': interaction_data.get('exercise_id', 'unknown'),
                    'code_submission': interaction_data.get('code_submission', ''),
                    'is_correct': interaction_data.get('is_correct', False),
                    'execution_time': interaction_data.get('execution_time', 0),
                    'attempts': interaction_data.get('attempts', 1)
                })
            
            elif processed_data['type'] == 'question_asked':
                processed_data.update({
                    'question_text': interaction_data.get('question_text', ''),
                    'question_category': interaction_data.get('question_category', 'general'),
                    'context': interaction_data.get('context', ''),
                    'urgency': interaction_data.get('urgency', 'normal')
                })
            
            elif processed_data['type'] == 'content_viewed':
                processed_data.update({
                    'content_id': interaction_data.get('content_id', 'unknown'),
                    'content_type': interaction_data.get('content_type', 'unknown'),
                    'view_duration': interaction_data.get('view_duration', 0),
                    'completion_percentage': interaction_data.get('completion_percentage', 0)
                })
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing interaction data: {e}")
            return {
                'interaction_id': f"error_{datetime.now().timestamp()}",
                'student_id': 'unknown',
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def aggregate_student_metrics(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics from student interactions.
        
        Args:
            interactions (list): List of interaction data
            
        Returns:
            dict: Aggregated metrics
        """
        try:
            if not interactions:
                return self._get_empty_metrics()
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(interactions)
            
            # Basic metrics
            metrics = {
                'total_interactions': len(interactions),
                'unique_days_active': len(df['timestamp'].dt.date.unique()) if 'timestamp' in df.columns else 0,
                'interaction_types': df['type'].value_counts().to_dict() if 'type' in df.columns else {},
                'avg_session_length': 0,  # Would need session data
                'last_activity': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
            
            # Code submission metrics
            code_submissions = df[df['type'] == 'code_submission'] if 'type' in df.columns else pd.DataFrame()
            if not code_submissions.empty:
                metrics['code_submission_metrics'] = {
                    'total_submissions': len(code_submissions),
                    'success_rate': code_submissions['is_correct'].mean() if 'is_correct' in code_submissions.columns else 0,
                    'avg_attempts': code_submissions['attempts'].mean() if 'attempts' in code_submissions.columns else 0,
                    'unique_exercises': code_submissions['exercise_id'].nunique() if 'exercise_id' in code_submissions.columns else 0
                }
            
            # Question metrics
            questions = df[df['type'] == 'question_asked'] if 'type' in df.columns else pd.DataFrame()
            if not questions.empty:
                metrics['question_metrics'] = {
                    'total_questions': len(questions),
                    'question_categories': questions['question_category'].value_counts().to_dict() if 'question_category' in questions.columns else {},
                    'avg_question_length': questions['question_text'].str.len().mean() if 'question_text' in questions.columns else 0
                }
            
            return metrics
            
        except Exception as e:
            print(f"Error aggregating student metrics: {e}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Get empty metrics structure."""
        return {
            'total_interactions': 0,
            'unique_days_active': 0,
            'interaction_types': {},
            'avg_session_length': 0,
            'last_activity': None,
            'code_submission_metrics': {
                'total_submissions': 0,
                'success_rate': 0,
                'avg_attempts': 0,
                'unique_exercises': 0
            },
            'question_metrics': {
                'total_questions': 0,
                'question_categories': {},
                'avg_question_length': 0
            }
        }
    
    def export_student_data(self, student_data: Dict[str, Any], format: str = 'json') -> str:
        """
        Export student data in specified format.
        
        Args:
            student_data (dict): Student data to export
            format (str): Export format ('json', 'csv')
            
        Returns:
            str: Exported data as string
        """
        try:
            if format.lower() == 'json':
                return json.dumps(student_data, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Flatten the data for CSV export
                flattened_data = self._flatten_dict(student_data)
                df = pd.DataFrame([flattened_data])
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            print(f"Error exporting student data: {e}")
            return ""
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d (dict): Dictionary to flatten
            parent_key (str): Parent key for nested items
            sep (str): Separator for nested keys
            
        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def import_student_data(self, data_string: str, format: str = 'json') -> Dict[str, Any]:
        """
        Import student data from string.
        
        Args:
            data_string (str): Data string to import
            format (str): Data format ('json', 'csv')
            
        Returns:
            dict: Imported student data
        """
        try:
            if format.lower() == 'json':
                data = json.loads(data_string)
                return self.validate_student_data(data)
            
            elif format.lower() == 'csv':
                df = pd.read_csv(pd.StringIO(data_string))
                if len(df) > 0:
                    data = df.iloc[0].to_dict()
                    return self.validate_student_data(data)
                else:
                    return self._get_default_student_data()
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
                
        except Exception as e:
            print(f"Error importing student data: {e}")
            return self._get_default_student_data()
