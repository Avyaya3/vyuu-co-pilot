#!/usr/bin/env python3
"""
Schema-based data extraction for financial data.

This module provides extraction functions for basic read operations
on financial data sent from NextJS, eliminating the need for MCP calls.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
import re

logger = logging.getLogger(__name__)


class FinancialDataExtractor:
    """
    Extracts financial data from request payload using schema-based approach.
    
    This class handles basic read operations for financial data without
    requiring external API calls or complex calculations.
    """
    
    def __init__(self, financial_data: Dict[str, Any]):
        """
        Initialize the extractor with financial data.
        
        Args:
            financial_data: The financial data from the request payload
        """
        self.data = financial_data
        self.user = financial_data.get("user", {})
        self.assets = financial_data.get("assets", [])
        self.liabilities = financial_data.get("liabilities", [])
        self.goals = financial_data.get("goals", [])
        self.income = financial_data.get("income", [])
        self.expenses = financial_data.get("expenses", [])
        self.stocks = financial_data.get("stocks", [])
        self.insurance = financial_data.get("insurance", [])
        self.savings = financial_data.get("savings", [])
        self.dashboard_metrics = financial_data.get("dashboardMetrics", {})
    
    def extract_assets(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract asset information based on filters.
        
        Args:
            filters: Optional filters to apply (category, min_value, max_value, etc.)
            
        Returns:
            Dictionary containing filtered assets and summary
        """
        filtered_assets = self.assets.copy()
        
        if filters:
            # Apply category filter
            if "category" in filters:
                filtered_assets = [
                    asset for asset in filtered_assets 
                    if asset.get("category", "").lower() == filters["category"].lower()
                ]
            
            # Apply subcategory filter
            if "subcategory" in filters:
                filtered_assets = [
                    asset for asset in filtered_assets 
                    if asset.get("subcategory", "").lower() == filters["subcategory"].lower()
                ]
            
            # Apply value filters
            if "min_value" in filters:
                filtered_assets = [
                    asset for asset in filtered_assets 
                    if asset.get("currentValue", 0) >= filters["min_value"]
                ]
            
            if "max_value" in filters:
                filtered_assets = [
                    asset for asset in filtered_assets 
                    if asset.get("currentValue", 0) <= filters["max_value"]
                ]
        
        # Calculate totals
        total_value = sum(asset.get("currentValue", 0) for asset in filtered_assets)
        total_purchase_value = sum(asset.get("purchaseValue", 0) for asset in filtered_assets)
        total_gain = total_value - total_purchase_value
        
        return {
            "assets": filtered_assets,
            "count": len(filtered_assets),
            "total_current_value": total_value,
            "total_purchase_value": total_purchase_value,
            "total_gain": total_gain,
            "total_gain_percentage": (total_gain / total_purchase_value * 100) if total_purchase_value > 0 else 0
        }
    
    def extract_liabilities(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract liability information based on filters.
        
        Args:
            filters: Optional filters to apply (type, min_amount, max_amount, etc.)
            
        Returns:
            Dictionary containing filtered liabilities and summary
        """
        filtered_liabilities = self.liabilities.copy()
        
        if filters:
            # Apply type filter
            if "type" in filters:
                filtered_liabilities = [
                    liability for liability in filtered_liabilities 
                    if liability.get("type", "").lower() == filters["type"].lower()
                ]
            
            # Apply amount filters
            if "min_amount" in filters:
                filtered_liabilities = [
                    liability for liability in filtered_liabilities 
                    if liability.get("amount", 0) >= filters["min_amount"]
                ]
            
            if "max_amount" in filters:
                filtered_liabilities = [
                    liability for liability in filtered_liabilities 
                    if liability.get("amount", 0) <= filters["max_amount"]
                ]
        
        # Calculate totals
        total_amount = sum(liability.get("amount", 0) for liability in filtered_liabilities)
        total_emi = sum(liability.get("emi", 0) for liability in filtered_liabilities)
        
        return {
            "liabilities": filtered_liabilities,
            "count": len(filtered_liabilities),
            "total_amount": total_amount,
            "total_emi": total_emi
        }
    
    def extract_income(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract income information based on filters.
        
        Args:
            filters: Optional filters to apply (source, frequency, category, etc.)
            
        Returns:
            Dictionary containing filtered income and summary
        """
        filtered_income = self.income.copy()
        
        if filters:
            # Apply source filter
            if "source" in filters:
                filtered_income = [
                    income for income in filtered_income 
                    if income.get("source", "").lower() == filters["source"].lower()
                ]
            
            # Apply frequency filter
            if "frequency" in filters:
                filtered_income = [
                    income for income in filtered_income 
                    if income.get("frequency", "").lower() == filters["frequency"].lower()
                ]
            
            # Apply category filter
            if "category" in filters:
                filtered_income = [
                    income for income in filtered_income 
                    if income.get("category", "").lower() == filters["category"].lower()
                ]
        
        # Calculate totals
        total_amount = sum(income.get("amount", 0) for income in filtered_income)
        
        # Group by frequency
        frequency_totals = {}
        for income in filtered_income:
            freq = income.get("frequency", "unknown")
            frequency_totals[freq] = frequency_totals.get(freq, 0) + income.get("amount", 0)
        
        return {
            "income": filtered_income,
            "count": len(filtered_income),
            "total_amount": total_amount,
            "frequency_totals": frequency_totals
        }
    
    def extract_expenses(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract expense information based on filters.
        
        Args:
            filters: Optional filters to apply (category, subcategory, min_amount, etc.)
            
        Returns:
            Dictionary containing filtered expenses and summary
        """
        filtered_expenses = self.expenses.copy()
        
        if filters:
            # Apply category filter
            if "category" in filters:
                filtered_expenses = [
                    expense for expense in filtered_expenses 
                    if expense.get("category", "").lower() == filters["category"].lower()
                ]
            
            # Apply subcategory filter
            if "subcategory" in filters:
                filtered_expenses = [
                    expense for expense in filtered_expenses 
                    if expense.get("subcategory", "").lower() == filters["subcategory"].lower()
                ]
            
            # Apply amount filters
            if "min_amount" in filters:
                filtered_expenses = [
                    expense for expense in filtered_expenses 
                    if expense.get("amount", 0) >= filters["min_amount"]
                ]
            
            if "max_amount" in filters:
                filtered_expenses = [
                    expense for expense in filtered_expenses 
                    if expense.get("amount", 0) <= filters["max_amount"]
                ]
        
        # Calculate totals
        total_amount = sum(expense.get("amount", 0) for expense in filtered_expenses)
        
        # Group by category
        category_totals = {}
        for expense in filtered_expenses:
            category = expense.get("category", "unknown")
            category_totals[category] = category_totals.get(category, 0) + expense.get("amount", 0)
        
        return {
            "expenses": filtered_expenses,
            "count": len(filtered_expenses),
            "total_amount": total_amount,
            "category_totals": category_totals
        }
    
    def extract_stocks(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract stock information based on filters.
        
        Args:
            filters: Optional filters to apply (type, subcategory, min_value, etc.)
            
        Returns:
            Dictionary containing filtered stocks and summary
        """
        filtered_stocks = self.stocks.copy()
        
        if filters:
            # Apply type filter
            if "type" in filters:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.get("type", "").lower() == filters["type"].lower()
                ]
            
            # Apply subcategory filter
            if "subcategory" in filters:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.get("subcategory", "").lower() == filters["subcategory"].lower()
                ]
            
            # Apply value filters
            if "min_value" in filters:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.get("currentValue", 0) >= filters["min_value"]
                ]
            
            if "max_value" in filters:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.get("currentValue", 0) <= filters["max_value"]
                ]
        
        # Calculate totals
        total_current_value = sum(stock.get("currentValue", 0) for stock in filtered_stocks)
        total_buy_value = sum(stock.get("buyValue", 0) for stock in filtered_stocks)
        total_gain = total_current_value - total_buy_value
        
        return {
            "stocks": filtered_stocks,
            "count": len(filtered_stocks),
            "total_current_value": total_current_value,
            "total_buy_value": total_buy_value,
            "total_gain": total_gain,
            "total_gain_percentage": (total_gain / total_buy_value * 100) if total_buy_value > 0 else 0
        }
    
    def extract_savings(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract savings information based on filters.
        
        Args:
            filters: Optional filters to apply (type, min_balance, etc.)
            
        Returns:
            Dictionary containing filtered savings and summary
        """
        filtered_savings = self.savings.copy()
        
        if filters:
            # Apply type filter
            if "type" in filters:
                filtered_savings = [
                    saving for saving in filtered_savings 
                    if saving.get("type", "").lower() == filters["type"].lower()
                ]
            
            # Apply balance filters
            if "min_balance" in filters:
                filtered_savings = [
                    saving for saving in filtered_savings 
                    if saving.get("currentBalance", 0) >= filters["min_balance"]
                ]
        
        # Calculate totals
        total_balance = sum(saving.get("currentBalance", 0) for saving in filtered_savings)
        total_monthly_contribution = sum(saving.get("monthlyContribution", 0) for saving in filtered_savings)
        
        return {
            "savings": filtered_savings,
            "count": len(filtered_savings),
            "total_balance": total_balance,
            "total_monthly_contribution": total_monthly_contribution
        }
    
    def extract_insurance(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract insurance information based on filters.
        
        Args:
            filters: Optional filters to apply (type, provider, min_coverage, etc.)
            
        Returns:
            Dictionary containing filtered insurance and summary
        """
        filtered_insurance = self.insurance.copy()
        
        if filters:
            # Apply type filter
            if "type" in filters:
                filtered_insurance = [
                    insurance for insurance in filtered_insurance 
                    if insurance.get("type", "").lower() == filters["type"].lower()
                ]
            
            # Apply provider filter
            if "provider" in filters:
                filtered_insurance = [
                    insurance for insurance in filtered_insurance 
                    if insurance.get("provider", "").lower() == filters["provider"].lower()
                ]
            
            # Apply coverage filters
            if "min_coverage" in filters:
                filtered_insurance = [
                    insurance for insurance in filtered_insurance 
                    if insurance.get("coverage", 0) >= filters["min_coverage"]
                ]
        
        # Calculate totals
        total_coverage = sum(insurance.get("coverage", 0) for insurance in filtered_insurance)
        total_premium = sum(insurance.get("premium", 0) for insurance in filtered_insurance)
        
        return {
            "insurance": filtered_insurance,
            "count": len(filtered_insurance),
            "total_coverage": total_coverage,
            "total_premium": total_premium
        }
    
    def extract_goals(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract financial goals information based on filters.
        
        Args:
            filters: Optional filters to apply (category, priority, etc.)
            
        Returns:
            Dictionary containing filtered goals and summary
        """
        filtered_goals = self.goals.copy()
        
        if filters:
            # Apply category filter
            if "category" in filters:
                filtered_goals = [
                    goal for goal in filtered_goals 
                    if goal.get("category", "").lower() == filters["category"].lower()
                ]
            
            # Apply priority filter
            if "priority" in filters:
                filtered_goals = [
                    goal for goal in filtered_goals 
                    if goal.get("priority", "").lower() == filters["priority"].lower()
                ]
        
        # Calculate totals
        total_target = sum(goal.get("target", 0) for goal in filtered_goals)
        total_current = sum(goal.get("current", 0) for goal in filtered_goals)
        total_remaining = total_target - total_current
        
        return {
            "goals": filtered_goals,
            "count": len(filtered_goals),
            "total_target": total_target,
            "total_current": total_current,
            "total_remaining": total_remaining,
            "completion_percentage": (total_current / total_target * 100) if total_target > 0 else 0
        }
    
    def get_net_worth(self) -> Dict[str, Any]:
        """
        Calculate net worth from assets and liabilities.
        
        Returns:
            Dictionary containing net worth calculation
        """
        # Use dashboard metrics if available, otherwise calculate
        if self.dashboard_metrics:
            return {
                "net_worth": self.dashboard_metrics.get("netWorth", 0),
                "total_assets": self.dashboard_metrics.get("totalAssets", 0),
                "total_liabilities": self.dashboard_metrics.get("totalLiabilities", 0),
                "source": "dashboard_metrics"
            }
        
        # Calculate from individual components
        total_assets = sum(asset.get("currentValue", 0) for asset in self.assets)
        total_assets += sum(stock.get("currentValue", 0) for stock in self.stocks)
        total_assets += sum(saving.get("currentBalance", 0) for saving in self.savings)
        
        total_liabilities = sum(liability.get("amount", 0) for liability in self.liabilities)
        
        net_worth = total_assets - total_liabilities
        
        return {
            "net_worth": net_worth,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "source": "calculated"
        }
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get dashboard metrics.
        
        Returns:
            Dictionary containing dashboard metrics
        """
        return self.dashboard_metrics.copy() if self.dashboard_metrics else {}
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get user information.
        
        Returns:
            Dictionary containing user information
        """
        return self.user.copy()
    
    def extract_by_query_type(self, query_type: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract data based on query type.
        
        Args:
            query_type: Type of data to extract (assets, liabilities, income, etc.)
            filters: Optional filters to apply
            
        Returns:
            Dictionary containing extracted data
        """
        query_type = query_type.lower()
        
        if query_type in ["assets", "asset"]:
            return self.extract_assets(filters)
        elif query_type in ["liabilities", "liability", "debt", "loans"]:
            return self.extract_liabilities(filters)
        elif query_type in ["income", "earnings", "salary"]:
            return self.extract_income(filters)
        elif query_type in ["expenses", "expense", "spending", "costs"]:
            return self.extract_expenses(filters)
        elif query_type in ["stocks", "stock", "equity", "investments"]:
            return self.extract_stocks(filters)
        elif query_type in ["savings", "saving", "deposits"]:
            return self.extract_savings(filters)
        elif query_type in ["insurance", "policies"]:
            return self.extract_insurance(filters)
        elif query_type in ["goals", "goal", "targets"]:
            return self.extract_goals(filters)
        elif query_type in ["net_worth", "networth", "wealth"]:
            return self.get_net_worth()
        elif query_type in ["dashboard", "metrics", "summary"]:
            return self.get_dashboard_metrics()
        elif query_type in ["user", "profile"]:
            return self.get_user_info()
        else:
            return {"error": f"Unknown query type: {query_type}"}


def create_data_extractor(financial_data: Dict[str, Any]) -> FinancialDataExtractor:
    """
    Create a FinancialDataExtractor instance.
    
    Args:
        financial_data: The financial data from the request payload
        
    Returns:
        FinancialDataExtractor instance
    """
    return FinancialDataExtractor(financial_data)
