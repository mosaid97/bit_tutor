# utilities/configuration/config_manager.py

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """
    Configuration manager for the BIT Tutor system.
    Handles loading, validation, and management of configuration settings.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir (str): Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config = {}
        self.default_config = self._get_default_config()
        print(f"ConfigManager initialized with config directory: {self.config_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'app': {
                'name': 'BIT Tutor',
                'version': '1.0.0',
                'debug': False,
                'host': '127.0.0.1',
                'port': 8080,
                'secret_key': 'your-secret-key-here'
            },
            'database': {
                'type': 'sqlite',
                'host': 'localhost',
                'port': 5432,
                'name': 'bit_tutor',
                'user': 'bit_tutor_user',
                'password': 'password',
                'connection_pool_size': 10
            },
            'neo4j': {
                'enabled': True,
                'uri': 'bolt://localhost:7687',
                'user': 'neo4j',
                'password': 'password',
                'database': 'neo4j'
            },
            'ai_models': {
                'knowledge_tracing': {
                    'model_type': 'MLFBK',
                    'embedding_dim': 64,
                    'num_heads': 8,
                    'num_layers': 4,
                    'max_seq_len': 100
                },
                'cognitive_diagnosis': {
                    'model_type': 'GNN_CDM',
                    'num_node_features': 4,
                    'embedding_dim': 64,
                    'learning_rate': 0.001
                },
                'recommendation': {
                    'model_type': 'RL_Agent',
                    'learning_rate': 0.1,
                    'discount_factor': 0.9,
                    'exploration_rate': 1.0,
                    'exploration_decay': 0.01
                }
            },
            'content_generation': {
                'llm_provider': 'openai',
                'model_name': 'gpt-3.5-turbo',
                'api_key': 'your-api-key-here',
                'max_tokens': 1000,
                'temperature': 0.7
            },
            'learning': {
                'default_difficulty': 'medium',
                'mastery_threshold': 0.7,
                'max_attempts_per_exercise': 5,
                'session_timeout_minutes': 30
            },
            'ui': {
                'theme': 'modern',
                'language': 'en',
                'items_per_page': 10,
                'enable_animations': True,
                'show_progress_bars': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/bit_tutor.log',
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'security': {
                'enable_csrf': True,
                'session_cookie_secure': True,
                'session_cookie_httponly': True,
                'password_min_length': 8,
                'max_login_attempts': 5
            }
        }
    
    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file (str): Configuration file name
            
        Returns:
            dict: Loaded configuration
        """
        config_path = self.config_dir / config_file
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        loaded_config = yaml.safe_load(f)
                    elif config_file.endswith('.json'):
                        loaded_config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {config_file}")
                
                # Merge with default config
                self.config = self._merge_configs(self.default_config, loaded_config)
                print(f"Configuration loaded from {config_path}")
            else:
                print(f"Config file {config_path} not found, using default configuration")
                self.config = self.default_config.copy()
                # Save default config to file
                self.save_config(config_file)
            
            # Override with environment variables
            self._load_env_overrides()
            
            return self.config
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            self.config = self.default_config.copy()
            return self.config
    
    def save_config(self, config_file: str = "config.yaml") -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_file (str): Configuration file name
            
        Returns:
            bool: True if successful, False otherwise
        """
        config_path = self.config_dir / config_file
        
        try:
            with open(config_path, 'w') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif config_file.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file}")
            
            print(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation, e.g., 'app.port')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            print(f"Error setting configuration value: {e}")
            return False
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded configuration with default configuration.
        
        Args:
            default (dict): Default configuration
            loaded (dict): Loaded configuration
            
        Returns:
            dict: Merged configuration
        """
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'BIT_TUTOR_DEBUG': 'app.debug',
            'BIT_TUTOR_HOST': 'app.host',
            'BIT_TUTOR_PORT': 'app.port',
            'BIT_TUTOR_SECRET_KEY': 'app.secret_key',
            'NEO4J_URI': 'neo4j.uri',
            'NEO4J_USER': 'neo4j.user',
            'NEO4J_PASSWORD': 'neo4j.password',
            'OPENAI_API_KEY': 'content_generation.api_key',
            'DATABASE_URL': 'database.url'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ('true', 'false'):
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                
                self.set(config_key, env_value)
                print(f"Configuration override from environment: {config_key} = {env_value}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate current configuration.
        
        Returns:
            dict: Validation results with errors and warnings
        """
        errors = []
        warnings = []
        
        # Validate required fields
        required_fields = [
            'app.name',
            'app.port',
            'app.secret_key'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                errors.append(f"Required field missing: {field}")
        
        # Validate data types and ranges
        if not isinstance(self.get('app.port'), int) or not (1 <= self.get('app.port') <= 65535):
            errors.append("app.port must be an integer between 1 and 65535")
        
        if self.get('app.secret_key') == 'your-secret-key-here':
            warnings.append("Using default secret key - please change for production")
        
        if self.get('content_generation.api_key') == 'your-api-key-here':
            warnings.append("Using default API key - please set your actual API key")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.
        
        Returns:
            dict: Configuration summary
        """
        return {
            'app_name': self.get('app.name'),
            'app_version': self.get('app.version'),
            'debug_mode': self.get('app.debug'),
            'host': self.get('app.host'),
            'port': self.get('app.port'),
            'neo4j_enabled': self.get('neo4j.enabled'),
            'ai_models_configured': len(self.get('ai_models', {})),
            'config_file_exists': (self.config_dir / "config.yaml").exists(),
            'total_config_keys': len(str(self.config).split(',')),
        }
