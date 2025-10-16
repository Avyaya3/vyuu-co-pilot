"""
Data formatting utilities for financial information.

This module provides comprehensive formatting functions for financial data
including currency, percentages, dates, transactions, and analysis results.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class DataFormatter:
    """
    Utility class for formatting financial data in a consistent, user-friendly manner.
    
    Features:
    - Currency formatting with symbols and commas
    - Percentage formatting with precision control
    - Date formatting in user-friendly formats
    - Transaction list formatting
    - Balance and account formatting
    - Spending analysis formatting
    - Simple text table generation
    """
    
    def __init__(self, currency_symbol: Optional[str] = None):
        """
        Initialize DataFormatter with optional currency symbol.
        
        Args:
            currency_symbol: Currency symbol to use. If None, uses default ₹.
        """
        self.currency_symbol = currency_symbol or "₹"

    def format_currency(self, amount: Union[int, float], currency_symbol: Optional[str] = None) -> str:
        """
        Format currency amount with symbol and commas.
        
        Args:
            amount: Amount to format
            currency_symbol: Currency symbol to use. If None, uses instance default.
            
        Returns:
            Formatted currency string
        """
        if currency_symbol is None:
            currency_symbol = self.currency_symbol
        if amount is None:
            return f"{currency_symbol}0.00"
        
        # Handle negative amounts
        is_negative = amount < 0
        abs_amount = abs(amount)
        
        # Format with commas and 2 decimal places
        formatted = f"{abs_amount:,.2f}"
        
        # Add currency symbol and handle negative
        if is_negative:
            return f"-{currency_symbol}{formatted}"
        else:
            return f"{currency_symbol}{formatted}"

    def format_percentage(self, value: Union[int, float], decimal_places: int = 1) -> str:
        """
        Format percentage with specified decimal places.
        
        Args:
            value: Percentage value (0-100)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if value is None:
            return "0%"
        
        return f"{value:.{decimal_places}f}%"

    def format_date(self, date_input: Union[str, datetime], format_type: str = "user_friendly") -> str:
        """
        Format date in user-friendly format.
        
        Args:
            date_input: Date string or datetime object
            format_type: Format type ("user_friendly", "short", "iso")
            
        Returns:
            Formatted date string
        """
        if not date_input:
            return "Unknown date"
        
        try:
            if isinstance(date_input, str):
                # Try to parse common date formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%d/%m/%Y"]:
                    try:
                        date_obj = datetime.strptime(date_input, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return str(date_input)  # Return as-is if can't parse
            else:
                date_obj = date_input
            
            if format_type == "user_friendly":
                return date_obj.strftime("%B %d, %Y")
            elif format_type == "short":
                return date_obj.strftime("%m/%d/%Y")
            elif format_type == "iso":
                return date_obj.strftime("%Y-%m-%d")
            else:
                return date_obj.strftime("%B %d, %Y")
                
        except Exception:
            return str(date_input)

    def format_transactions(self, transactions: List[Dict[str, Any]], max_items: int = 10) -> str:
        """
        Format list of transactions in a readable format.
        
        Args:
            transactions: List of transaction dictionaries
            max_items: Maximum number of items to show
            
        Returns:
            Formatted transaction list
        """
        if not transactions:
            return "No transactions found."
        
        formatted_items = []
        for i, tx in enumerate(transactions[:max_items]):
            amount = tx.get("amount", 0)
            description = tx.get("description", "Unknown")
            date = tx.get("date", "Unknown date")
            
            formatted_amount = self.format_currency(amount)
            formatted_date = self.format_date(date, "short")
            
            formatted_items.append(f"{formatted_amount} - {description} ({formatted_date})")
        
        if len(transactions) > max_items:
            formatted_items.append(f"... and {len(transactions) - max_items} more")
        
        return "\n".join(formatted_items)

    def format_accounts(self, accounts: List[Dict[str, Any]], max_items: int = 5) -> str:
        """
        Format list of accounts in a readable format.
        
        Args:
            accounts: List of account dictionaries
            max_items: Maximum number of items to show
            
        Returns:
            Formatted account list
        """
        if not accounts:
            return "No accounts found."
        
        formatted_items = []
        for account in accounts[:max_items]:
            name = account.get("name", "Unknown")
            account_type = account.get("type", "Unknown type")
            balance = account.get("balance", 0)
            
            formatted_balance = self.format_currency(balance)
            formatted_items.append(f"{name} ({account_type}): {formatted_balance}")
        
        if len(accounts) > max_items:
            formatted_items.append(f"... and {len(accounts) - max_items} more")
        
        return "\n".join(formatted_items)

    def format_balances(self, balances: Dict[str, Any]) -> str:
        """
        Format account balances in a readable format.
        
        Args:
            balances: Dictionary of account balances
            
        Returns:
            Formatted balance information
        """
        if not balances:
            return "No balance information available."
        
        formatted_items = []
        for account, balance in balances.items():
            if isinstance(balance, (int, float)):
                formatted_balance = self.format_currency(balance)
                formatted_items.append(f"{account}: {formatted_balance}")
        
        return "\n".join(formatted_items)

    def format_spending_analysis(self, spending_data: Dict[str, Any]) -> str:
        """
        Format spending analysis data.
        
        Args:
            spending_data: Spending analysis dictionary
            
        Returns:
            Formatted spending analysis
        """
        if not spending_data:
            return "No spending analysis available."
        
        formatted_items = []
        for category, amount in spending_data.items():
            if isinstance(amount, (int, float)):
                formatted_amount = self.format_currency(amount)
                formatted_items.append(f"{category}: {formatted_amount}")
        
        return "\n".join(formatted_items)

    def format_monthly_summary(self, summary_data: Dict[str, Any]) -> str:
        """
        Format monthly summary data.
        
        Args:
            summary_data: Monthly summary dictionary
            
        Returns:
            Formatted monthly summary
        """
        if not summary_data:
            return "No monthly summary available."
        
        formatted_items = []
        for month, data in summary_data.items():
            if isinstance(data, dict):
                income = data.get("income", 0)
                expenses = data.get("expenses", 0)
                
                formatted_income = self.format_currency(income)
                formatted_expenses = self.format_currency(expenses)
                
                formatted_items.append(f"{month}: Income {formatted_income}, Expenses {formatted_expenses}")
        
        return "\n".join(formatted_items)

    def format_simple_table(self, data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
        """
        Format data as a simple text table.
        
        Args:
            data: List of dictionaries to format
            headers: Optional list of headers to use
            
        Returns:
            Formatted table string
        """
        if not data:
            return "No data available."
        
        if not headers:
            headers = list(data[0].keys()) if data else []
        
        # Calculate column widths
        col_widths = []
        for header in headers:
            max_width = len(str(header))
            for row in data:
                value = str(row.get(header, ""))
                max_width = max(max_width, len(value))
            col_widths.append(max_width)
        
        # Create header row
        header_row = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
        separator = "-" * len(header_row)
        
        # Create data rows
        data_rows = []
        for row in data:
            row_str = " | ".join(f"{str(row.get(header, '')):<{width}}" for header, width in zip(headers, col_widths))
            data_rows.append(row_str)
        
        return f"{header_row}\n{separator}\n" + "\n".join(data_rows)

    def format_data_for_llm(self, data: Dict[str, Any]) -> str:
        """
        Format data in a way that's optimized for LLM consumption.
        
        Args:
            data: Data dictionary to format
            
        Returns:
            Formatted string for LLM
        """
        if not data:
            return "No data available."
        
        formatted_parts = []
        
        # Handle balance data
        if "balances" in data or "balance" in data:
            balance_data = data.get("balances", data.get("balance", {}))
            if balance_data:
                formatted_parts.append("**Account Balances:**")
                for account, balance in balance_data.items():
                    if isinstance(balance, (int, float)):
                        formatted_parts.append(f"- {account}: {self.format_currency(balance)}")
                formatted_parts.append("")

        # Handle transaction data
        if "transactions" in data:
            transactions = data["transactions"]
            if transactions:
                formatted_parts.append(f"**Recent Transactions ({len(transactions)} found):**")
                for i, tx in enumerate(transactions[:5]):  # Show first 5
                    amount = tx.get("amount", 0)
                    description = tx.get("description", "Unknown")
                    date = tx.get("date", "Unknown date")
                    formatted_parts.append(
                        f"- {self.format_currency(amount)} - {description} ({self.format_date(date, 'short')})"
                    )
                formatted_parts.append("")

        # Handle account data
        if "accounts" in data:
            accounts = data["accounts"]
            if accounts:
                formatted_parts.append(f"**Accounts ({len(accounts)} found):**")
                for account in accounts[:3]:  # Show first 3
                    name = account.get("name", "Unknown")
                    account_type = account.get("type", "Unknown type")
                    balance = account.get("balance", 0)
                    formatted_parts.append(f"- {name} ({account_type}): {self.format_currency(balance)}")
                formatted_parts.append("")

        # Handle spending analysis
        if "spending_analysis" in data:
            spending = data["spending_analysis"]
            if spending:
                formatted_parts.append("**Spending Analysis:**")
                for category, amount in spending.items():
                    if isinstance(amount, (int, float)):
                        formatted_parts.append(f"- {category}: {self.format_currency(amount)}")
                formatted_parts.append("")

        # Handle monthly summary
        if "monthly_summary" in data:
            summary = data["monthly_summary"]
            if summary:
                formatted_parts.append("**Monthly Summary:**")
                for month, month_data in summary.items():
                    if isinstance(month_data, dict):
                        income = month_data.get("income", 0)
                        expenses = month_data.get("expenses", 0)
                        formatted_parts.append(
                            f"- {month}: Income {self.format_currency(income)}, "
                            f"Expenses {self.format_currency(expenses)}"
                        )
                formatted_parts.append("")

        # Handle any other data types
        for key, value in data.items():
            if key not in ["balances", "balance", "transactions", "accounts", "spending_analysis", "monthly_summary"]:
                if key == "advice":
                    # Special handling for advice data - preserve the full text
                    if isinstance(value, dict) and "advice" in value:
                        formatted_parts.append(value["advice"])
                    elif isinstance(value, str):
                        formatted_parts.append(value)
                    else:
                        formatted_parts.append(f"**{key.title()}:** {value}")
                elif isinstance(value, dict):
                    formatted_parts.append(f"**{key.title()}:**")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            formatted_parts.append(f"- {sub_key}: {self.format_currency(sub_value)}")
                        else:
                            formatted_parts.append(f"- {sub_key}: {sub_value}")
                    formatted_parts.append("")
                elif isinstance(value, list):
                    formatted_parts.append(f"**{key.title()} ({len(value)} items):**")
                    for item in value[:3]:  # Show first 3
                        formatted_parts.append(f"- {item}")
                    formatted_parts.append("")
                else:
                    formatted_parts.append(f"**{key.title()}:** {value}")
                    formatted_parts.append("")

        return "\n".join(formatted_parts).strip()
