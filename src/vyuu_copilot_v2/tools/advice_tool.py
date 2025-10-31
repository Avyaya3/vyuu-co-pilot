"""
Financial advice tool that provides personalized recommendations using LLM.

This tool analyzes user queries and financial data context to provide
personalized financial advice and recommendations.
"""

import re
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
    
    def _safe_get_number(self, data: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely get a number from a dictionary, handling None values."""
        value = data.get(key, default)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_format_currency(self, value: Any, default: float = 0.0) -> str:
        """Safely format a value as currency."""
        if value is None:
            value = default
        try:
            return f"₹{float(value):,}"
        except (ValueError, TypeError):
            return f"₹{default:,}"
    
    def _safe_format_percentage(self, value: Any, default: float = 0.0) -> str:
        """Safely format a value as percentage."""
        if value is None:
            value = default
        try:
            return f"{float(value) * 100:.1f}%"
        except (ValueError, TypeError):
            return f"{default * 100:.1f}%"
    
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
        
        # Parse the advice response to separate recommendations from calculations
        parsed_advice = self._parse_advice_response(advice_response)
        
        return {
            "advice": parsed_advice["recommendations"],  # Clean recommendations without calculations
            "calculations": parsed_advice["calculations"],  # Mathematical calculations separately
            "full_response": advice_response,  # Keep full response for backward compatibility
            "user_query": params.user_query,
            "context_used": bool(params.context_data or params.financial_data),
            "user_id": params.user_id,
            "risk_levels": ["high", "medium", "low"],
            "has_calculations": True
        }
    
    def _build_system_prompt(self) -> str:
        """Build an optimized system prompt for three-level risk advice."""
        return """You are a financial advisor. Provide THREE separate advice recommendations based on different risk levels.

STRUCTURE REQUIREMENTS (CRITICAL):
1. Provide ALL recommendations FIRST (for all three risk levels)
2. Then provide ALL mathematical calculations in a SEPARATE section AFTER the recommendations
3. Calculations must NEVER be inline with recommendations - they must be in a completely separate section

For each risk level (HIGH RISK, MEDIUM RISK, LOW RISK):
1. Provide 3-5 specific, actionable recommendations (bullet points only, NO calculations here)
2. Use ₹ for amounts in recommendations

After all three risk level recommendations, provide a SEPARATE "Mathematical Calculations:" section that includes:
- Step-by-step breakdown of calculations for each risk level
- Key assumptions and formulas used
- Expected outcomes with numbers
- Show the working (e.g., ₹50,000 × (1 + 0.12) = ₹50,000 × 1.12 = ₹56,000)

CRITICAL FORMATTING RULES:
- ALL recommendations must come BEFORE any calculations
- Calculations section must start with a clear header: "Mathematical Calculations:" or "### Mathematical Calculations:"
- Do NOT include calculations inline with recommendations
- Do NOT put calculations under each individual risk level section
- Put all calculations together in one separate section at the end

Example CORRECT format:
## HIGH RISK Strategy
### Recommendations:
- Recommendation 1
- Recommendation 2

## MEDIUM RISK Strategy
### Recommendations:
- Recommendation 1
- Recommendation 2

## LOW RISK Strategy
### Recommendations:
- Recommendation 1
- Recommendation 2

### Mathematical Calculations:
- [All calculations for all risk levels here]

Let the advice strategy vary based on the user's question and their risk tolerance. Consider:
- High Risk: More aggressive growth strategies, higher equity allocation
- Medium Risk: Balanced approach, mix of equity and debt
- Low Risk: Conservative strategies, capital preservation focus

Format each risk level clearly with recommendations first, then all calculations separately."""
    
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
                prompt_parts.append(f"- Name: {user.get('name') or 'N/A'}")
                monthly_income = self._safe_get_number(user, 'monthly_income', 0)
                monthly_expenses = self._safe_get_number(user, 'monthly_expenses', 0)
                prompt_parts.append(f"- Monthly Income: {self._safe_format_currency(monthly_income)}")
                prompt_parts.append(f"- Monthly Expenses: {self._safe_format_currency(monthly_expenses)}")
                prompt_parts.append(f"- Risk Profile: {user.get('risk_profile') or 'N/A'}")
                prompt_parts.append("")
            
            # Dashboard metrics
            if 'dashboardMetrics' in financial_data:
                metrics = financial_data['dashboardMetrics']
                prompt_parts.append(f"**Financial Overview:**")
                prompt_parts.append(f"- Net Worth: {self._safe_format_currency(metrics.get('netWorth'))}")
                prompt_parts.append(f"- Current Savings Rate: {self._safe_format_percentage(metrics.get('savingsRate'))}")
                prompt_parts.append(f"- Total Assets: {self._safe_format_currency(metrics.get('totalAssets'))}")
                prompt_parts.append(f"- Total Liabilities: {self._safe_format_currency(metrics.get('totalLiabilities'))}")
                prompt_parts.append("")
            
            # Assets
            if 'assets' in financial_data and financial_data['assets']:
                prompt_parts.append(f"**Assets:**")
                for asset in financial_data['assets']:
                    asset_name = asset.get('name') or 'N/A'
                    asset_value = self._safe_format_currency(asset.get('currentValue'))
                    asset_category = asset.get('category') or 'N/A'
                    prompt_parts.append(f"- {asset_name}: {asset_value} ({asset_category})")
                prompt_parts.append("")
            
            # Liabilities
            if 'liabilities' in financial_data and financial_data['liabilities']:
                prompt_parts.append(f"**Liabilities:**")
                for liability in financial_data['liabilities']:
                    liability_name = liability.get('name') or 'N/A'
                    liability_amount = self._safe_format_currency(liability.get('amount'))
                    liability_emi = self._safe_format_currency(liability.get('emi'))
                    prompt_parts.append(f"- {liability_name}: {liability_amount} (EMI: {liability_emi})")
                prompt_parts.append("")
            
            # Goals
            if 'goals' in financial_data and financial_data['goals']:
                prompt_parts.append(f"**Financial Goals:**")
                for goal in financial_data['goals']:
                    goal_name = goal.get('name') or 'N/A'
                    current = self._safe_get_number(goal, 'current', 0)
                    target = self._safe_get_number(goal, 'target', 0)
                    progress = (current / target * 100) if target > 0 else 0
                    prompt_parts.append(f"- {goal_name}: {self._safe_format_currency(current)} / {self._safe_format_currency(target)} ({progress:.1f}% complete)")
                prompt_parts.append("")
            
            prompt_parts.append("=" * 50)
            
            # Add calculation-ready summary
            prompt_parts.append("\n**Key Metrics for Calculations:**")
            
            # Calculate disposable income if possible
            user = financial_data.get('user', {})
            monthly_income = self._safe_get_number(user, 'monthly_income', 0)
            monthly_expenses = self._safe_get_number(user, 'monthly_expenses', 0)
            disposable_income = monthly_income - monthly_expenses
            
            metrics = financial_data.get('dashboardMetrics', {})
            net_worth = self._safe_get_number(metrics, 'netWorth', 0)
            
            prompt_parts.append(f"- Monthly Disposable Income: {self._safe_format_currency(disposable_income)}")
            prompt_parts.append(f"- Current Net Worth: {self._safe_format_currency(net_worth)}")
            prompt_parts.append(f"- Current Savings Rate: {self._safe_format_percentage(metrics.get('savingsRate'))}")
            
            # Add user risk profile if available
            risk_profile = user.get('risk_profile') or 'moderate'
            prompt_parts.append(f"- User's Risk Profile: {risk_profile}")
            prompt_parts.append("")
            
        elif params.context_data:
            prompt_parts.append(f"\nFinancial Context:\n{params.context_data}")
        else:
            prompt_parts.append("\nNote: No specific financial data context was provided.")
        
        # Add request for three-level advice with calculations
        prompt_parts.append("""
Provide THREE separate advice recommendations for the user's question, one for each risk level.

CRITICAL FORMATTING REQUIREMENTS:
1. Provide ALL recommendations FIRST (for all three risk levels) - NO calculations in this section
2. Then provide ALL mathematical calculations in a SEPARATE section at the END
3. Use this exact structure:

## HIGH RISK Strategy
### Recommendations:
- [3-5 bullet point recommendations - NO calculations here]

## MEDIUM RISK Strategy
### Recommendations:
- [3-5 bullet point recommendations - NO calculations here]

## LOW RISK Strategy
### Recommendations:
- [3-5 bullet point recommendations - NO calculations here]

### Mathematical Calculations:
[ALL calculations for ALL risk levels go here - separate section at the end]
- Show step-by-step calculation with formulas for each risk level
- Include all working (e.g., ₹50,000 × 1.12 = ₹56,000)
- Show expected returns/outcomes with full calculation breakdown
- Label calculations by risk level (HIGH RISK Calculations:, MEDIUM RISK Calculations:, LOW RISK Calculations:)

IMPORTANT: 
- Do NOT include calculations inline with recommendations
- Do NOT put calculations under each individual risk level
- Put ALL calculations together in the separate "Mathematical Calculations:" section at the end
- For each expected outcome, you MUST show the complete mathematical working. Do not just state the final number - show how you arrived at it.

Use their financial data (disposable income: ₹X, net worth: ₹Y) to make calculations specific and personalized.""")
        
        return "\n".join(prompt_parts)
    
    def _parse_advice_response(self, advice_response: str) -> Dict[str, str]:
        """
        Parse the advice response to separate recommendations from calculations.
        
        PRESERVES MARKDOWN FORMATTING by using regex-based extraction instead of
        splitting and stripping, which would break markdown structure.
        
        Args:
            advice_response: Full advice response from LLM
            
        Returns:
            Dictionary with 'recommendations' and 'calculations' keys, both
            preserving original markdown formatting
        """
        try:
            # Use regex to find calculation sections WITHOUT breaking markdown
            # Look for various patterns that indicate calculations section
            calc_patterns = [
                r'(?i)(\n\s*Mathematical\s+Calculations?:?\s*\n)',
                r'(?i)(\n\s*Calculations?:?\s*\n)',
                r'(?i)(\n\s*###?\s*Calculations?\s*\n)',
            ]
            
            calc_match = None
            calc_pattern_used = None
            
            # Try each pattern to find where calculations section starts
            for pattern in calc_patterns:
                match = re.search(pattern, advice_response)
                if match:
                    calc_match = match
                    calc_pattern_used = pattern
                    break
            
            if calc_match:
                # Found calculations section
                calc_start_pos = calc_match.start()
                calc_header = calc_match.group(1)
                
                # Everything before calculations = recommendations
                # Use rstrip() to only remove trailing whitespace, preserving leading/formatting
                recommendations_text = advice_response[:calc_start_pos].rstrip()
                
                # Everything from calculations header onwards = calculations
                # Use lstrip() only on the header itself to preserve rest of formatting
                calculations_text = advice_response[calc_start_pos:].lstrip()
                
                self.logger.debug(f"Successfully parsed advice response: found calculations section at position {calc_start_pos}")
                
                return {
                    "recommendations": recommendations_text,  # Preserves all markdown formatting
                    "calculations": calculations_text  # Preserves all markdown formatting
                }
            else:
                # No clear calculations section found
                # Check if calculations might be embedded within recommendations
                # (e.g., each risk level has its own calculations)
                if any(keyword in advice_response.lower() for keyword in ['calculation', 'formula', '×', '=', '%']):
                    # Likely has calculations but not in a separate section
                    # Return full response as recommendations, but try to extract what we can
                    self.logger.debug("No separate calculations section found, but calculations appear to be embedded")
                    return {
                        "recommendations": advice_response,  # Full response with embedded calculations
                        "calculations": ""  # Empty since calculations are embedded in recommendations
                    }
                else:
                    # No calculations detected at all
                    self.logger.debug("No calculations section found in advice response")
                    return {
                        "recommendations": advice_response,
                        "calculations": ""
                    }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse advice response: {e}")
            # On error, preserve the original markdown by returning full response
            return {
                "recommendations": advice_response,
                "calculations": ""
            }


# Create tool instance
advice_tool = AdviceTool()
