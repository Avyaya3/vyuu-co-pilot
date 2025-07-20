#!/usr/bin/env python3
"""
Demo script for Clarification Question Generator Node.

This script demonstrates the functionality of the clarification question generator
node with various scenarios and edge cases.
"""

import asyncio
import logging
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the node and related components
from src.schemas.state_schemas import ClarificationState, IntentType
from src.nodes.clarification_question_generator_node import (
    clarification_question_generator_node,
    ClarificationQuestionGenerator,
    QuestionTemplate
)


async def demo_basic_question_generation():
    """Demo basic question generation."""
    logger.info("=" * 60)
    logger.info("Demo: Basic Question Generation")
    logger.info("=" * 60)
    
    # Create a basic clarification state
    state = ClarificationState(
        user_input="Transfer money",
        session_id="demo-session-001",
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["amount", "account"],
        missing_critical_params=["amount"],
        parameter_priorities=["amount", "account"],
        clarification_attempts=0,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "How much would you like to transfer?"
        mock_llm_class.return_value = mock_llm
        
        # Generate question
        result_state = await clarification_question_generator_node(state)
        
        # Display results
        logger.info(f"Generated Question: {result_state.pending_question}")
        logger.info(f"Waiting for Response: {result_state.waiting_for_response}")
        logger.info(f"Clarification Phase: {result_state.clarification_phase}")
        logger.info(f"Attempts: {result_state.clarification_attempts}")
        logger.info(f"Status: {result_state.metadata.get('clarification_status')}")
        
        # Verify state updates
        assert result_state.pending_question == "How much would you like to transfer?"
        assert result_state.waiting_for_response is True
        assert result_state.clarification_phase == "waiting"
        assert result_state.clarification_attempts == 1
        
        logger.info("✅ Basic question generation successful!")


async def demo_question_with_extracted_parameters():
    """Demo question generation with some already extracted parameters."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Question with Extracted Parameters")
    logger.info("=" * 60)
    
    # Create state with some extracted parameters
    state = ClarificationState(
        user_input="Transfer $500",
        session_id="demo-session-002",
        intent=IntentType.ACTION,
        confidence=0.9,
        messages=[],
        metadata={},
        missing_params=["account"],
        missing_critical_params=["account"],
        extracted_parameters={"amount": "$500"},
        parameter_priorities=["account"],
        clarification_attempts=0,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "Which account would you like to transfer from?"
        mock_llm_class.return_value = mock_llm
        
        result_state = await clarification_question_generator_node(state)
        
        logger.info(f"Generated Question: {result_state.pending_question}")
        logger.info(f"Extracted Parameters: {result_state.extracted_parameters}")
        logger.info(f"Missing Parameters: {result_state.missing_params}")
        
        assert result_state.pending_question == "Which account would you like to transfer from?"
        assert result_state.extracted_parameters["amount"] == "$500"
        
        logger.info("✅ Question with extracted parameters successful!")


async def demo_max_attempts_reached():
    """Demo behavior when max attempts are reached."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Max Attempts Reached")
    logger.info("=" * 60)
    
    # Create state with max attempts reached
    state = ClarificationState(
        user_input="Transfer money",
        session_id="demo-session-003",
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["amount"],
        missing_critical_params=["amount"],
        clarification_attempts=3,  # Max attempts reached
        max_attempts=3
    )
    
    result_state = await clarification_question_generator_node(state)
    
    logger.info(f"Status: {result_state.metadata.get('clarification_status')}")
    logger.info(f"Exit Message: {result_state.metadata.get('exit_message')}")
    logger.info(f"Remaining Missing: {result_state.metadata.get('remaining_missing_params')}")
    
    assert result_state.metadata["clarification_status"] == "max_attempts_reached"
    assert "exit_message" in result_state.metadata
    
    logger.info("✅ Max attempts reached handling successful!")


async def demo_llm_failure_fallback():
    """Demo fallback behavior when LLM fails."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: LLM Failure Fallback")
    logger.info("=" * 60)
    
    state = ClarificationState(
        user_input="Transfer money",
        session_id="demo-session-004",
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["amount"],
        missing_critical_params=["amount"],
        clarification_attempts=0,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        # Mock LLM failure
        mock_llm = AsyncMock()
        mock_llm.chat_completion.side_effect = Exception("LLM API error")
        mock_llm_class.return_value = mock_llm
        
        result_state = await clarification_question_generator_node(state)
        
        logger.info(f"Status: {result_state.metadata.get('clarification_status')}")
        logger.info(f"Fallback Question: {result_state.pending_question}")
        logger.info(f"Error: {result_state.metadata.get('error')}")
        
        assert result_state.metadata["clarification_status"] == "error_fallback"
        assert result_state.pending_question is not None
        assert "error" in result_state.metadata
        
        logger.info("✅ LLM failure fallback successful!")


async def demo_clarification_history():
    """Demo question generation with existing clarification history."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Clarification History")
    logger.info("=" * 60)
    
    # Create state with existing history
    state = ClarificationState(
        user_input="Transfer money",
        session_id="demo-session-005",
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["account"],
        missing_critical_params=["account"],
        parameter_priorities=["amount", "account"],
        clarification_history=[
            {
                "question": "How much would you like to transfer?",
                "user_response": "$500",
                "targeted_param": "amount",
                "attempt": 1
            }
        ],
        clarification_attempts=1,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "Which account would you like to transfer from?"
        mock_llm_class.return_value = mock_llm
        
        result_state = await clarification_question_generator_node(state)
        
        logger.info(f"Generated Question: {result_state.pending_question}")
        logger.info(f"History Length: {len(result_state.clarification_history)}")
        logger.info(f"Attempts: {result_state.clarification_attempts}")
        
        # Should ask about account (not amount, since it was already asked)
        assert "account" in result_state.pending_question.lower()
        assert len(result_state.clarification_history) == 2
        assert result_state.clarification_attempts == 2
        
        logger.info("✅ Clarification history handling successful!")


async def demo_different_intent_types():
    """Demo question generation for different intent types."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Different Intent Types")
    logger.info("=" * 60)
    
    # Test DATA_FETCH intent
    data_fetch_state = ClarificationState(
        user_input="Show me transactions",
        session_id="demo-session-006",
        intent=IntentType.DATA_FETCH,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["entity_type"],
        missing_critical_params=["entity_type"],
        clarification_attempts=0,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "What type of data would you like to see?"
        mock_llm_class.return_value = mock_llm
        
        result_state = await clarification_question_generator_node(data_fetch_state)
        
        logger.info(f"DATA_FETCH Question: {result_state.pending_question}")
        assert "data" in result_state.pending_question.lower() or "type" in result_state.pending_question.lower()
    
    # Test AGGREGATE intent
    aggregate_state = ClarificationState(
        user_input="Summarize spending",
        session_id="demo-session-007",
        intent=IntentType.AGGREGATE,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["metric_type"],
        missing_critical_params=["metric_type"],
        clarification_attempts=0,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "What kind of analysis would you like?"
        mock_llm_class.return_value = mock_llm
        
        result_state = await clarification_question_generator_node(aggregate_state)
        
        logger.info(f"AGGREGATE Question: {result_state.pending_question}")
        assert "analysis" in result_state.pending_question.lower() or "kind" in result_state.pending_question.lower()
    
    logger.info("✅ Different intent types handling successful!")


async def demo_question_template_functionality():
    """Demo the QuestionTemplate functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Question Template Functionality")
    logger.info("=" * 60)
    
    state = ClarificationState(
        user_input="Transfer about five hundred",
        session_id="demo-session-008",
        intent=IntentType.ACTION,
        confidence=0.9,
        messages=[],
        metadata={},
        missing_params=["source_account"],
        missing_critical_params=["source_account"],
        extracted_parameters={"amount": 500.0},
        normalization_suggestions={
            "amount": "User said 'about' - confirm exact amount"
        },
        ambiguity_flags={
            "amount": "approximation"
        },
        clarification_attempts=0,
        max_attempts=3
    )
    
    # Test context prompt building
    context_prompt = QuestionTemplate.build_context_prompt(state)
    logger.info("Context Prompt:")
    logger.info(context_prompt)
    
    # Test instruction prompt building
    instruction_prompt = QuestionTemplate.build_instruction_prompt(
        ["source_account"],
        state.normalization_suggestions,
        state.ambiguity_flags,
        state.intent
    )
    logger.info("Instruction Prompt:")
    logger.info(instruction_prompt)
    
    # Test intent guidance
    guidance = QuestionTemplate._get_intent_guidance(IntentType.ACTION, ["source_account"])
    logger.info(f"Intent Guidance: {guidance}")
    
    logger.info("✅ Question template functionality successful!")


async def main():
    """Run all demos."""
    logger.info("Starting Clarification Question Generator Demos")
    logger.info("=" * 80)
    
    try:
        await demo_basic_question_generation()
        await demo_question_with_extracted_parameters()
        await demo_max_attempts_reached()
        await demo_llm_failure_fallback()
        await demo_clarification_history()
        await demo_different_intent_types()
        await demo_question_template_functionality()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 All demos completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 