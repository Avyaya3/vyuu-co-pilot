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
from .asset_repository import AssetRepository
from .liability_repository import LiabilityRepository
from .savings_repository import SavingsRepository
from .income_repository import IncomeRepository
from .expense_repository import ExpenseRepository
from .stock_repository import StockRepository
from .insurance_repository import InsuranceRepository
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
            self._repositories['asset'] = AssetRepository()
            self._repositories['liability'] = LiabilityRepository()
            self._repositories['savings'] = SavingsRepository()
            self._repositories['income'] = IncomeRepository()
            self._repositories['expense'] = ExpenseRepository()
            self._repositories['stock'] = StockRepository()
            self._repositories['insurance'] = InsuranceRepository()
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
    
    def get_asset_repository(self) -> AssetRepository:
        """Get the asset repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['asset']
    
    def get_liability_repository(self) -> LiabilityRepository:
        """Get the liability repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['liability']
    
    def get_savings_repository(self) -> SavingsRepository:
        """Get the savings repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['savings']
    
    def get_income_repository(self) -> IncomeRepository:
        """Get the income repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['income']
    
    def get_expense_repository(self) -> ExpenseRepository:
        """Get the expense repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['expense']
    
    def get_stock_repository(self) -> StockRepository:
        """Get the stock repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['stock']
    
    def get_insurance_repository(self) -> InsuranceRepository:
        """Get the insurance repository instance."""
        if not self._initialized:
            self.initialize()
        return self._repositories['insurance']
    
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


def get_asset_repository() -> AssetRepository:
    """Get asset repository instance."""
    return get_repository_factory().get_asset_repository()


def get_liability_repository() -> LiabilityRepository:
    """Get liability repository instance."""
    return get_repository_factory().get_liability_repository()


def get_savings_repository() -> SavingsRepository:
    """Get savings repository instance."""
    return get_repository_factory().get_savings_repository()


def get_income_repository() -> IncomeRepository:
    """Get income repository instance."""
    return get_repository_factory().get_income_repository()


def get_expense_repository() -> ExpenseRepository:
    """Get expense repository instance."""
    return get_repository_factory().get_expense_repository()


def get_stock_repository() -> StockRepository:
    """Get stock repository instance."""
    return get_repository_factory().get_stock_repository()


def get_insurance_repository() -> InsuranceRepository:
    """Get insurance repository instance."""
    return get_repository_factory().get_insurance_repository()


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
    'AssetRepository', 
    'LiabilityRepository',
    'SavingsRepository',
    'IncomeRepository',
    'ExpenseRepository',
    'StockRepository',
    'InsuranceRepository',
    'GoalRepository',
    
    # Factory and convenience functions
    'RepositoryFactory',
    'get_repository_factory',
    'get_user_repository',
    'get_asset_repository',
    'get_liability_repository',
    'get_savings_repository',
    'get_income_repository',
    'get_expense_repository',
    'get_stock_repository',
    'get_insurance_repository',
    'get_goal_repository',
] 