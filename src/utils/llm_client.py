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
from typing import Dict, Any, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Simple centralized LLM client for all node operations.
    
    This client provides a configured OpenAI client that nodes can use directly.
    Each node is responsible for building its own prompts and handling its own
    business logic.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4-turbo", 
        temperature: float = 0.2,
        max_tokens: int = 1500,
        timeout: int = 30 ):
        """
        Initialize LLM client with configuration.
        
        Args:
            model: OpenAI model to use (default: gpt-4)
            temperature: Temperature for generation (default: 0.2)
            max_tokens: Maximum tokens per response (default: 1500)
            timeout: Request timeout in seconds (default: 30)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        logger.info(f"LLM client initialized: model={model}, temperature={temperature}")
    
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
            
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content.strip()
            
            logger.debug(f"LLM response received: length={len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"LLM chat completion failed: {e}")
            raise 