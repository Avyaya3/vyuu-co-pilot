"""
Centralized LLM Client for Vyuu Copilot v2

This module provides a simple, centralized LLM client for all nodes that need to interact
with language models. It handles OpenAI connection, configuration, and basic error handling.

Features:
- Centralized configuration and credentials
- Standardized model settings 
- Basic error handling and logging
- Simple interface for nodes to use
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Optimized centralized LLM client for all node operations.
    
    This client provides task-specific model selection and optimized configurations
    for different types of operations to maximize performance while maintaining quality.
    """
    
    # Task-specific model configurations for optimal performance
    MODEL_CONFIGS = {
        "intent_classification": {
            "model": "gpt-4-turbo",  # Keep high accuracy for classification
            "temperature": 0.1,
            "max_tokens": 200,
            "timeout": 15
        },
        "parameter_extraction": {
            "model": "gpt-3.5-turbo",  # Faster for parameter extraction
            "temperature": 0.1,
            "max_tokens": 300,
            "timeout": 15
        },
        "execution_planning": {
            "model": "gpt-3.5-turbo",  # Faster for planning
            "temperature": 0.2,
            "max_tokens": 400,
            "timeout": 15
        },
        "advice_generation": {
            "model": "gpt-3.5-turbo",  # Faster for advice
            "temperature": 0.3,
            "max_tokens": 600,
            "timeout": 20
        },
        "response_synthesis": {
            "model": "gpt-3.5-turbo",  # Faster for synthesis
            "temperature": 0.3,
            "max_tokens": 800,
            "timeout": 20
        },
        "default": {
            "model": "gpt-3.5-turbo",  # Default to faster model
            "temperature": 0.2,
            "max_tokens": 500,
            "timeout": 15
        }
    }
    
    def __init__(
        self, 
        task_type: str = "default",
        model: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None ):
        """
        Initialize LLM client with task-specific optimization.
        
        Args:
            task_type: Type of task for optimized configuration
            model: Override model (optional)
            temperature: Override temperature (optional)
            max_tokens: Override max_tokens (optional)
            timeout: Override timeout (optional)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Get task-specific configuration
        config = self.MODEL_CONFIGS.get(task_type, self.MODEL_CONFIGS["default"])
        
        # Use overrides if provided, otherwise use task-specific defaults
        self.model = model or config["model"]
        self.temperature = temperature if temperature is not None else config["temperature"]
        self.max_tokens = max_tokens if max_tokens is not None else config["max_tokens"]
        self.timeout = timeout if timeout is not None else config["timeout"]
        self.task_type = task_type
        
        self.client = OpenAI(api_key=api_key, timeout=self.timeout)
        
        logger.info(f"LLM client initialized: task={task_type}, model={self.model}, "
                   f"temperature={self.temperature}, max_tokens={self.max_tokens}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs ) -> str:
        """
        Simple chat completion method for nodes to use.
        
        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            response_format: Response format specification
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Response content string
            
        Raises:
            Exception: If LLM call fails
        """
        try:
            # Use instance defaults if not overridden
            call_temperature = temperature if temperature is not None else self.temperature
            call_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": call_temperature,
                "max_tokens": call_max_tokens,
                **kwargs
            }
            
            # Add response format if specified
            if response_format:
                request_params["response_format"] = response_format
            
            logger.debug(f"Making LLM request: model={self.model}, messages={len(messages)}")
            
            # Use asyncio.to_thread to make the blocking OpenAI call async
            def _make_openai_call():
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content
            
            content = await asyncio.to_thread(_make_openai_call)
            
            logger.debug(f"LLM response received: length={len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"LLM chat completion failed: {e}")
            raise 

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs ) -> str:
        """
        Generate a response using system and user prompts.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Generated response string
            
        Raises:
            Exception: If LLM call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    @classmethod
    def for_task(cls, task_type: str, **kwargs) -> "LLMClient":
        """
        Create an LLM client optimized for a specific task type.
        
        Args:
            task_type: Type of task (intent_classification, parameter_extraction, etc.)
            **kwargs: Additional configuration overrides
            
        Returns:
            Optimized LLMClient instance
        """
        return cls(task_type=task_type, **kwargs) 