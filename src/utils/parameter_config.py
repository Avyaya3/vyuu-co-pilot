"""
Parameter Configuration Manager for Intent Router.

This module provides configuration-driven parameter definitions for intent routing,
replacing hardcoded parameter dictionaries with YAML-based configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Set
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class ParameterConfigManager:
    """
    Manages parameter configuration for intent routing decisions.
    
    Loads intent parameter definitions from YAML configuration and provides
    methods to access critical and optional parameters for each intent type.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the parameter configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default to config file in project root
            config_path = Path(__file__).parent.parent.parent / "config" / "intent_parameters.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
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
        return set(critical_params)
    
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
        return set(optional_params)
    
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


def get_parameter_config() -> ParameterConfigManager:
    """
    Get the global parameter configuration manager instance.
    
    Returns:
        Singleton ParameterConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
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