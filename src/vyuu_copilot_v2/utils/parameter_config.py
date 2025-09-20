"""
Parameter Configuration Manager for Intent Parameters.

This module provides a centralized way to manage intent parameter configurations
from YAML files, replacing the hardcoded parameter dictionaries with a flexible
configuration system.

Features:
- YAML-based configuration files
- Support for critical and optional parameters
- Intent descriptions and metadata
- Backward compatibility with legacy format
- Caching for performance
- Error handling and fallbacks
"""

import logging
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Set, List, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class ParameterConfigManager:
    """
    Manages intent parameter configurations from YAML files.
    
    This class provides a flexible way to define and access parameter configurations
    for different intents, supporting both critical and optional parameters with
    descriptions and metadata.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the parameter configuration manager.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to config/intent_parameters.yaml
            self.config_path = Path("config/intent_parameters.yaml")
        
        self._config_data = None
        
        # Check if we're in an async context and avoid file reading
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use fallback config
            self._config_data = self._get_fallback_config()
            logger.info("Using fallback config in async context")
        except RuntimeError:
            # Not in async context, safe to read file
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # Simple synchronous file read - this should be fine since it's called in __init__
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file)
                
            if not self._config_data or 'intent_parameters' not in self._config_data:
                raise ValueError("Invalid configuration format: missing 'intent_parameters' section")
                
            logger.info(f"Loaded parameter configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load parameter configuration: {e}")
            # Fall back to empty config rather than failing
            self._config_data = {'intent_parameters': {}}
            raise
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration when file reading is not possible."""
        return {
            'intent_parameters': {
                'data_fetch': {
                    'critical': ['user_id'],
                    'optional': ['account_name', 'account_id', 'limit', 'days_back'],
                    'description': 'Fetch financial data from database'
                },
                'aggregate': {
                    'critical': ['user_id'],
                    'optional': ['account_name', 'account_id', 'start_date', 'end_date', 'days_back'],
                    'description': 'Perform financial aggregations and analysis'
                },
                'action': {
                    'critical': ['user_id'],
                    'optional': ['account_name', 'account_id', 'amount', 'transaction_type'],
                    'description': 'Perform financial actions like transfers and transactions'
                }
            }
        }
    
    @lru_cache(maxsize=32)
    def get_critical_parameters(self, intent: str) -> Set[str]:
        """
        Get critical parameters for an intent.
        
        Args:
            intent: Intent name (e.g., 'data_fetch', 'aggregate', 'action')
            
        Returns:
            Set of critical parameter names for the intent
        """
        intent_config = self._config_data.get('intent_parameters', {}).get(intent, {})
        critical_params = intent_config.get('critical', [])
        
        # Handle both old format (list of strings) and new format (list of dicts)
        param_names = []
        for param in critical_params:
            if isinstance(param, str):
                param_names.append(param)
            elif isinstance(param, dict):
                param_names.append(param.get('name', ''))
        
        return set(filter(None, param_names))  # Filter out empty strings
    
    @lru_cache(maxsize=32)
    def get_optional_parameters(self, intent: str) -> Set[str]:
        """
        Get optional parameters for an intent.
        
        Args:
            intent: Intent name (e.g., 'data_fetch', 'aggregate', 'action')
            
        Returns:
            Set of optional parameter names for the intent
        """
        intent_config = self._config_data.get('intent_parameters', {}).get(intent, {})
        optional_params = intent_config.get('optional', [])
        
        # Handle both old format (list of strings) and new format (list of dicts)
        param_names = []
        for param in optional_params:
            if isinstance(param, str):
                param_names.append(param)
            elif isinstance(param, dict):
                param_names.append(param.get('name', ''))
        
        return set(filter(None, param_names))  # Filter out empty strings
    
    @lru_cache(maxsize=32)
    def get_all_parameters(self, intent: str) -> Set[str]:
        """
        Get all (critical + optional) parameters for an intent.
        
        Args:
            intent: Intent name
            
        Returns:
            Set of all parameter names for the intent
        """
        return self.get_critical_parameters(intent) | self.get_optional_parameters(intent)
    
    def get_intent_description(self, intent: str) -> str:
        """
        Get description for an intent.
        
        Args:
            intent: Intent name
            
        Returns:
            Human-readable description of the intent
        """
        intent_config = self._config_data.get('intent_parameters', {}).get(intent, {})
        return intent_config.get('description', f"No description available for {intent}")
    
    def get_available_intents(self) -> List[str]:
        """
        Get list of all configured intents.
        
        Returns:
            List of intent names configured in the system
        """
        return list(self._config_data.get('intent_parameters', {}).keys())
    
    def to_legacy_format(self) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Convert configuration to legacy format for backward compatibility.
        
        Returns:
            Tuple of (critical_parameters_dict, optional_parameters_dict) matching
            the original hardcoded dictionary format
        """
        intents = self.get_available_intents()
        
        critical_params = {}
        optional_params = {}
        
        for intent in intents:
            critical_params[intent] = self.get_critical_parameters(intent)
            optional_params[intent] = self.get_optional_parameters(intent)
        
        return critical_params, optional_params


# Global instance for easy access
_config_manager = None
_config_lock = asyncio.Lock()


def get_parameter_config() -> ParameterConfigManager:
    """
    Get the global parameter configuration manager instance.
    
    Returns:
        Singleton ParameterConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        # Initialize with a simple fallback config to avoid blocking
        _config_manager = ParameterConfigManager()
    return _config_manager


async def get_parameter_config_async() -> ParameterConfigManager:
    """
    Get the global parameter configuration manager instance asynchronously.
    
    Returns:
        Singleton ParameterConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        async with _config_lock:
            if _config_manager is None:
                # Initialize with a simple fallback config to avoid blocking
                _config_manager = ParameterConfigManager()
    return _config_manager


def reload_parameter_config(config_path: str = None) -> ParameterConfigManager:
    """
    Reload parameter configuration with optional new path.
    
    Args:
        config_path: Optional new configuration file path
        
    Returns:
        New ParameterConfigManager instance
    """
    global _config_manager
    _config_manager = ParameterConfigManager(config_path)
    return _config_manager 