"""
Repository Layer Initialization and Factory.

This module provides centralized access to all repositories with dependency injection
support and factory patterns for easy testing and configuration.

Features:
- Repository factory for dependency injection
- Centralized repository management
- Easy testing with mock repositories
- Health check aggregation
"""

import logging
from typing import Dict, Any, Optional

from .base_repository import BaseRepository, RepositoryError, EntityNotFoundError, EntityValidationError, DatabaseOperationError
from .user_repository import UserRepository
from .account_repository import AccountRepository
from .transaction_repository import TransactionRepository
from .goal_repository import GoalRepository

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """
    Factory class for creating and managing repository instances.
    
    Provides centralized access to all repositories with singleton pattern
    and dependency injection support.
    """
    
    def __init__(self):
        """Initialize the repository factory."""
        self._repositories: Dict[str, BaseRepository] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all repositories."""
        if self._initialized:
            return
        
        try:
            self._repositories['user'] = UserRepository()
            self._repositories['account'] = AccountRepository()
            self._repositories['transaction'] = TransactionRepository()
            self._repositories['goal'] = GoalRepository()
            
            self._initialized = True
            logger.info("Repository factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize repository factory: {e}")
            raise RepositoryError(f"Repository factory initialization failed: {e}")
    
    def get_user_repository(self) -> UserRepository:
        """Get the user repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['user']
    
    def get_account_repository(self) -> AccountRepository:
        """Get the account repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['account']
    
    def get_transaction_repository(self) -> TransactionRepository:
        """Get the transaction repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['transaction']
    
    def get_goal_repository(self) -> GoalRepository:
        """Get the goal repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['goal']
    
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Perform health checks on all repositories.
        
        Returns:
            Aggregated health check results
        """
        if not self._initialized:
            self.initialize()
        
        health_results = {
            'status': 'unknown',
            'repositories': {},
            'summary': {
                'total': len(self._repositories),
                'healthy': 0,
                'unhealthy': 0
            }
        }
        
        for name, repo in self._repositories.items():
            try:
                repo_health = await repo.health_check()
                health_results['repositories'][name] = repo_health
                
                if repo_health.get('status') == 'healthy':
                    health_results['summary']['healthy'] += 1
                else:
                    health_results['summary']['unhealthy'] += 1
                    
            except Exception as e:
                health_results['repositories'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_results['summary']['unhealthy'] += 1
        
        # Determine overall status
        if health_results['summary']['unhealthy'] == 0:
            health_results['status'] = 'healthy'
        elif health_results['summary']['healthy'] > 0:
            health_results['status'] = 'degraded'
        else:
            health_results['status'] = 'unhealthy'
        
        return health_results
    
    def reset(self) -> None:
        """Reset the factory (useful for testing)."""
        self._repositories.clear()
        self._initialized = False
        logger.info("Repository factory reset")


# Global repository factory instance
_repository_factory: Optional[RepositoryFactory] = None


def get_repository_factory() -> RepositoryFactory:
    """
    Get the global repository factory instance.
    
    Returns:
        Repository factory instance
    """
    global _repository_factory
    if _repository_factory is None:
        _repository_factory = RepositoryFactory()
    return _repository_factory


# Convenience functions for direct repository access
def get_user_repository() -> UserRepository:
    """Get user repository instance."""
    return get_repository_factory().get_user_repository()


def get_account_repository() -> AccountRepository:
    """Get account repository instance."""
    return get_repository_factory().get_account_repository()


def get_transaction_repository() -> TransactionRepository:
    """Get transaction repository instance."""
    return get_repository_factory().get_transaction_repository()


def get_goal_repository() -> GoalRepository:
    """Get goal repository instance."""
    return get_repository_factory().get_goal_repository()


# Export all repository classes and exceptions
__all__ = [
    # Base classes and exceptions
    'BaseRepository',
    'RepositoryError',
    'EntityNotFoundError', 
    'EntityValidationError',
    'DatabaseOperationError',
    
    # Repository classes
    'UserRepository',
    'AccountRepository', 
    'TransactionRepository',
    'GoalRepository',
    
    # Factory and convenience functions
    'RepositoryFactory',
    'get_repository_factory',
    'get_user_repository',
    'get_account_repository',
    'get_transaction_repository',
    'get_goal_repository',
] 