"""
Financial advice tool that provides personalized recommendations using LLM.

This tool analyzes user queries and financial data context to provide
personalized financial advice and recommendations.
"""

import time
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .base import ToolResponse, ToolInterface
from vyuu_copilot_v2.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AdviceParams(BaseModel):
    """
    Parameters for financial advice operations.
    """
    
    user_query: str = Field(description="The user's question or request for advice")
    context_data: Optional[str] = Field(None, description="Financial data context provided from frontend")
    user_id: Optional[str] = Field(None, description="User ID for personalized advice")
    financial_data: Optional[Dict[str, Any]] = Field(None, description="Complete financial data from user's profile")


class AdviceTool(ToolInterface):
    """
    Tool for providing personalized financial advice using LLM.
    """
    
    name: str = "advice"
    schema = AdviceParams
    
    def __init__(self):
        """Initialize the advice tool with optimized LLM client."""
        self.logger = logging.getLogger(__name__)
        try:
            # Use task-specific optimized LLM client for advice generation
            self.llm_client = LLMClient.for_task("advice_generation")
        except ValueError as e:
            self.logger.warning(f"LLM client initialization failed: {e}")
            self.llm_client = None
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide financial advice based on user query and context.
        
        Args:
            params: Dictionary of parameters matching AdviceParams schema
            
        Returns:
            Dictionary with advice result
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            validated_params = AdviceParams(**params)
            self.logger.info(f"Providing advice for user query: {validated_params.user_query[:100]}...")
            
            # Generate advice using LLM
            advice_result = await self._generate_advice(validated_params)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "data": advice_result,
                "tool_name": self.name,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Advice generation failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name,
                "execution_time_ms": execution_time
            }
    
    async def _generate_advice(self, params: AdviceParams) -> Dict[str, Any]:
        """Generate personalized financial advice using LLM."""
        
        if self.llm_client is None:
            return {
                "advice": "LLM client not available. Please configure OPENAI_API_KEY environment variable.",
                "user_query": params.user_query,
                "context_used": bool(params.context_data),
                "user_id": params.user_id,
                "error": "LLM client not initialized"
            }
        
        # Build the system prompt
        system_prompt = self._build_system_prompt()
        
        # Build the user prompt with context
        user_prompt = self._build_user_prompt(params)
        
        # Call LLM for advice using optimized settings
        # Uses task-specific optimized settings: gpt-3.5-turbo, temperature=0.3, max_tokens=600
        advice_response = await self.llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return {
            "advice": advice_response,
            "user_query": params.user_query,
            "context_used": bool(params.context_data),
            "user_id": params.user_id
        }
    
    def _build_system_prompt(self) -> str:
        """Build an optimized, concise system prompt for financial advice."""
        return """You are a financial advisor. Provide concise, actionable advice in 3-5 bullet points.

Guidelines:
- Be specific and actionable
- Use the user's financial data when available
- Prioritize security and realistic goals
- Keep advice practical and implementable

Focus on: budgeting, savings, investments, debt management, and financial planning."""
    
    def _build_user_prompt(self, params: AdviceParams) -> str:
        """Build the user prompt with query and context."""
        prompt_parts = []
        
        # Add the user's question
        prompt_parts.append(f"User Question: {params.user_query}")
        
        # Add user ID if available
        if params.user_id:
            prompt_parts.append(f"\nUser ID: {params.user_id}")
        
        # Add financial data if available
        if params.financial_data:
            prompt_parts.append("\nUser's Financial Profile:")
            prompt_parts.append("=" * 50)
            
            # Format the financial data nicely
            financial_data = params.financial_data
            
            # User basic info
            if 'user' in financial_data:
                user = financial_data['user']
                prompt_parts.append(f"**Personal Information:**")
                prompt_parts.append(f"- Name: {user.get('name', 'N/A')}")
                prompt_parts.append(f"- Monthly Income: ₹{user.get('monthly_income', 0):,}")
                prompt_parts.append(f"- Monthly Expenses: ₹{user.get('monthly_expenses', 0):,}")
                prompt_parts.append(f"- Risk Profile: {user.get('risk_profile', 'N/A')}")
                prompt_parts.append("")
            
            # Dashboard metrics
            if 'dashboardMetrics' in financial_data:
                metrics = financial_data['dashboardMetrics']
                prompt_parts.append(f"**Financial Overview:**")
                prompt_parts.append(f"- Net Worth: ₹{metrics.get('netWorth', 0):,}")
                prompt_parts.append(f"- Current Savings Rate: {metrics.get('savingsRate', 0)*100:.1f}%")
                prompt_parts.append(f"- Total Assets: ₹{metrics.get('totalAssets', 0):,}")
                prompt_parts.append(f"- Total Liabilities: ₹{metrics.get('totalLiabilities', 0):,}")
                prompt_parts.append("")
            
            # Assets
            if 'assets' in financial_data and financial_data['assets']:
                prompt_parts.append(f"**Assets:**")
                for asset in financial_data['assets']:
                    prompt_parts.append(f"- {asset.get('name', 'N/A')}: ₹{asset.get('currentValue', 0):,} ({asset.get('category', 'N/A')})")
                prompt_parts.append("")
            
            # Liabilities
            if 'liabilities' in financial_data and financial_data['liabilities']:
                prompt_parts.append(f"**Liabilities:**")
                for liability in financial_data['liabilities']:
                    prompt_parts.append(f"- {liability.get('name', 'N/A')}: ₹{liability.get('amount', 0):,} (EMI: ₹{liability.get('emi', 0):,})")
                prompt_parts.append("")
            
            # Goals
            if 'goals' in financial_data and financial_data['goals']:
                prompt_parts.append(f"**Financial Goals:**")
                for goal in financial_data['goals']:
                    current = goal.get('current', 0)
                    target = goal.get('target', 0)
                    progress = (current / target * 100) if target > 0 else 0
                    prompt_parts.append(f"- {goal.get('name', 'N/A')}: ₹{current:,} / ₹{target:,} ({progress:.1f}% complete)")
                prompt_parts.append("")
            
            prompt_parts.append("=" * 50)
        elif params.context_data:
            prompt_parts.append(f"\nFinancial Context:\n{params.context_data}")
        else:
            prompt_parts.append("\nNote: No specific financial data context was provided.")
        
        # Add request for concise advice
        prompt_parts.append("""
Provide specific, actionable financial advice in 3-5 bullet points. Use their financial data and provide concrete next steps.""")
        
        return "\n".join(prompt_parts)


# Create tool instance
advice_tool = AdviceTool()
