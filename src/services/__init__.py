"""
Services package for business logic operations.

This package provides high-level business logic services that orchestrate
repository operations and implement domain-specific business rules.
"""

import logging
from typing import Optional

# from .financial_service import FinancialService  # Temporarily disabled due to repository changes

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating and managing service instances.
    
    Provides centralized service instantiation and lifecycle management.
    Ensures singleton pattern for services within application context.
    """
    
    def __init__(self):
        self._services: dict = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize all services with their dependencies."""
        if self._initialized:
            return
        
        # Services are initialized lazily when requested
        self._initialized = True
        logger.info("Service factory initialized")
    
    # def get_financial_service(self) -> FinancialService:  # Temporarily disabled
    #     """
    #     Get financial service instance.
    #     
    #     Returns:
    #         FinancialService instance for financial operations
    #     """
    #     if 'financial_service' not in self._services:
    #         self._services['financial_service'] = FinancialService()
    #         logger.debug("Created new FinancialService instance")
    #     
    #     return self._services['financial_service']
    
    def reset(self):
        """Reset all services (useful for testing)."""
        self._services.clear()
        self._initialized = False
        logger.info("Service factory reset")


# Global service factory instance
_service_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """
    Get the global service factory instance.
    
    Returns:
        Service factory instance
    """
    global _service_factory
    if _service_factory is None:
        _service_factory = ServiceFactory()
    return _service_factory


# Convenience functions for direct service access
# def get_financial_service() -> FinancialService:  # Temporarily disabled
#     """Get financial service instance."""
#     return get_service_factory().get_financial_service()


# Export commonly used items
__all__ = [
    "ServiceFactory",
    "get_service_factory", 
    # "get_financial_service",  # Temporarily disabled
    # "FinancialService"  # Temporarily disabled
] 