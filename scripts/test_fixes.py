#!/usr/bin/env python3
"""
Test script to verify fixes for parameter parsing inconsistencies and state conversion edge cases.

This script tests:
1. Parameter parsing normalization in missing_param_analysis_node
2. State reconstruction in wrapper nodes
3. Edge case handling for AddableValuesDict and other state types
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the modules we're testing
from src.schemas.state_schemas import MainState, ClarificationState, IntentType, MessageRole
from src.nodes.missing_param_analysis_node import (
    normalize_parameter_priorities, 
    normalize_missing_params,
    missing_param_analysis_node
)
from src.subgraphs.clarification_subgraph import (
    reconstruct_state_from_dict,
    clarification_entry_wrapper,
    clarification_exit_wrapper
)
from src.subgraphs.direct_orchestrator_subgraph import (
    orchestrator_entry_wrapper,
    orchestrator_exit_wrapper
)


def test_parameter_priorities_normalization():
    """Test the parameter_priorities normalization function."""
    logger.info("Testing parameter_priorities normalization...")
    
    # Test cases
    test_cases = [
        # Case 1: Already a list (should work as is)
        {
            "input": ["entity_type", "amount", "account"],
            "expected": ["entity_type", "amount", "account"],
            "description": "List of strings"
        },
        # Case 2: Dict with numeric priorities
        {
            "input": {"entity_type": 3, "amount": 1, "account": 2},
            "expected": ["entity_type", "account", "amount"],  # Sorted by priority (high to low)
            "description": "Dict with numeric priorities"
        },
        # Case 3: Dict with string priorities
        {
            "input": {"entity_type": "high", "amount": "low", "account": "medium"},
            "expected": ["entity_type", "amount", "account"],  # Insertion order
            "description": "Dict with string priorities"
        },
        # Case 4: Single string
        {
            "input": "entity_type",
            "expected": ["entity_type"],
            "description": "Single string"
        },
        # Case 5: Empty/None values
        {
            "input": ["entity_type", None, "", "amount"],
            "expected": ["entity_type", "amount"],
            "description": "List with None/empty values"
        },
        # Case 6: Unknown type
        {
            "input": 123,
            "expected": [],
            "description": "Unknown type (number)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = normalize_parameter_priorities(test_case["input"])
            expected = test_case["expected"]
            
            if result == expected:
                logger.info(f"✅ Test {i} PASSED: {test_case['description']}")
            else:
                logger.error(f"❌ Test {i} FAILED: {test_case['description']}")
                logger.error(f"   Expected: {expected}")
                logger.error(f"   Got: {result}")
                
        except Exception as e:
            logger.error(f"❌ Test {i} ERROR: {test_case['description']} - {e}")
    
    logger.info("Parameter priorities normalization tests completed.\n")


def test_missing_params_normalization():
    """Test the missing_params normalization function."""
    logger.info("Testing missing_params normalization...")
    
    # Test cases
    test_cases = [
        # Case 1: Already a list
        {
            "input": ["entity_type", "amount"],
            "expected": ["entity_type", "amount"],
            "description": "List of strings"
        },
        # Case 2: Single string
        {
            "input": "entity_type",
            "expected": ["entity_type"],
            "description": "Single string"
        },
        # Case 3: Empty/None values
        {
            "input": ["entity_type", None, "", "amount"],
            "expected": ["entity_type", "amount"],
            "description": "List with None/empty values"
        },
        # Case 4: Unknown type
        {
            "input": {"entity_type": "high"},
            "expected": [],
            "description": "Unknown type (dict)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = normalize_missing_params(test_case["input"])
            expected = test_case["expected"]
            
            if result == expected:
                logger.info(f"✅ Test {i} PASSED: {test_case['description']}")
            else:
                logger.error(f"❌ Test {i} FAILED: {test_case['description']}")
                logger.error(f"   Expected: {expected}")
                logger.error(f"   Got: {result}")
                
        except Exception as e:
            logger.error(f"❌ Test {i} ERROR: {test_case['description']} - {e}")
    
    logger.info("Missing params normalization tests completed.\n")


def test_state_reconstruction():
    """Test the state reconstruction function."""
    logger.info("Testing state reconstruction...")
    
    # Create a sample MainState with proper UUID
    original_state = MainState(
        user_input="Show me my balance",
        session_id=str(uuid4()),
        intent=IntentType.DATA_FETCH,
        confidence=0.95,
        messages=[],
        metadata={"test": "data"}
    )
    
    # Test cases
    test_cases = [
        # Case 1: Already correct type
        {
            "input": original_state,
            "target_class": MainState,
            "description": "Already MainState object"
        },
        # Case 2: Dict representation
        {
            "input": original_state.model_dump(),
            "target_class": MainState,
            "description": "Dict representation"
        },
        # Case 3: AddableValuesDict-like object
        {
            "input": type('MockAddableValuesDict', (), {
                '__dict__': original_state.model_dump(),
                '__iter__': lambda self: iter(self.__dict__.items()),
                'items': lambda self: self.__dict__.items()
            })(),
            "target_class": MainState,
            "description": "AddableValuesDict-like object"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = reconstruct_state_from_dict(test_case["input"], test_case["target_class"])
            
            # Verify the reconstructed state has the same key attributes
            if (isinstance(result, test_case["target_class"]) and
                result.user_input == original_state.user_input and
                result.session_id == original_state.session_id and
                result.intent == original_state.intent):
                logger.info(f"✅ Test {i} PASSED: {test_case['description']}")
            else:
                logger.error(f"❌ Test {i} FAILED: {test_case['description']}")
                logger.error(f"   Expected type: {test_case['target_class']}")
                logger.error(f"   Got type: {type(result)}")
                
        except Exception as e:
            logger.error(f"❌ Test {i} ERROR: {test_case['description']} - {e}")
    
    logger.info("State reconstruction tests completed.\n")


async def test_wrapper_nodes():
    """Test the wrapper nodes with various state types."""
    logger.info("Testing wrapper nodes...")
    
    # Create a sample MainState with proper UUID
    main_state = MainState(
        user_input="Transfer money",
        session_id=str(uuid4()),
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={"user_id": "test_user"}
    )
    
    # Test clarification entry wrapper
    try:
        logger.info("Testing clarification_entry_wrapper...")
        clarification_state = await clarification_entry_wrapper(main_state)
        
        if isinstance(clarification_state, ClarificationState):
            logger.info("✅ clarification_entry_wrapper PASSED: Proper state conversion")
        else:
            logger.error(f"❌ clarification_entry_wrapper FAILED: Expected ClarificationState, got {type(clarification_state)}")
            
    except Exception as e:
        logger.error(f"❌ clarification_entry_wrapper ERROR: {e}")
    
    # Test with dict input (simulating AddableValuesDict)
    try:
        logger.info("Testing clarification_entry_wrapper with dict input...")
        dict_state = main_state.model_dump()
        clarification_state = await clarification_entry_wrapper(dict_state)
        
        if isinstance(clarification_state, ClarificationState):
            logger.info("✅ clarification_entry_wrapper with dict PASSED: State reconstruction worked")
        else:
            logger.error(f"❌ clarification_entry_wrapper with dict FAILED: Expected ClarificationState, got {type(clarification_state)}")
            
    except Exception as e:
        logger.error(f"❌ clarification_entry_wrapper with dict ERROR: {e}")
    
    # Test orchestrator entry wrapper
    try:
        logger.info("Testing orchestrator_entry_wrapper...")
        orchestrator_state = await orchestrator_entry_wrapper(main_state)
        
        if hasattr(orchestrator_state, 'extracted_params'):  # Check if it's OrchestratorState
            logger.info("✅ orchestrator_entry_wrapper PASSED: Proper state conversion")
        else:
            logger.error(f"❌ orchestrator_entry_wrapper FAILED: Expected OrchestratorState, got {type(orchestrator_state)}")
            
    except Exception as e:
        logger.error(f"❌ orchestrator_entry_wrapper ERROR: {e}")
    
    logger.info("Wrapper node tests completed.\n")


async def test_missing_param_analysis_with_normalization():
    """Test the missing_param_analysis_node with LLM responses that need normalization."""
    logger.info("Testing missing_param_analysis_node with normalization...")
    
    # Create a sample ClarificationState with proper UUID
    clarification_state = ClarificationState(
        user_input="Transfer money",
        session_id=str(uuid4()),
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["amount", "account"],
        missing_critical_params=["amount"]
    )
    
    # Mock LLM responses that need normalization
    mock_responses = [
        # Response 1: parameter_priorities as dict
        {
            "extracted_parameters": {"amount": None, "account": None},
            "missing_params": ["amount", "account"],
            "missing_critical_params": ["amount"],
            "parameter_priorities": {"amount": 1, "account": 2},  # Dict instead of list
            "normalization_suggestions": {},
            "ambiguity_flags": {}
        },
        # Response 2: parameter_priorities as string
        {
            "extracted_parameters": {"amount": None, "account": None},
            "missing_params": "amount",  # String instead of list
            "missing_critical_params": ["amount"],
            "parameter_priorities": "amount",  # String instead of list
            "normalization_suggestions": {},
            "ambiguity_flags": {}
        }
    ]
    
    for i, mock_response in enumerate(mock_responses, 1):
        try:
            with patch('src.nodes.missing_param_analysis_node.LLMClient') as mock_llm:
                # Mock the LLM response
                mock_client = AsyncMock()
                mock_client.chat_completion.return_value = json.dumps(mock_response)
                mock_llm.return_value = mock_client
                
                # Call the node
                result = await missing_param_analysis_node(clarification_state)
                
                # Check that parameter_priorities is a list
                if isinstance(result.parameter_priorities, list):
                    logger.info(f"✅ Test {i} PASSED: parameter_priorities normalized to list")
                    logger.info(f"   Result: {result.parameter_priorities}")
                else:
                    logger.error(f"❌ Test {i} FAILED: parameter_priorities not normalized")
                    logger.error(f"   Type: {type(result.parameter_priorities)}")
                    logger.error(f"   Value: {result.parameter_priorities}")
                    
        except Exception as e:
            logger.error(f"❌ Test {i} ERROR: {e}")
    
    logger.info("Missing param analysis normalization tests completed.\n")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Parameter Parsing and State Conversion Fixes")
    logger.info("=" * 60)
    
    # Run synchronous tests
    test_parameter_priorities_normalization()
    test_missing_params_normalization()
    test_state_reconstruction()
    
    # Run asynchronous tests
    await test_wrapper_nodes()
    await test_missing_param_analysis_with_normalization()
    
    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main()) 