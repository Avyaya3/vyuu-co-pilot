#!/usr/bin/env python3
"""
Demo script for Pause/Resume Clarification Flow.

This script demonstrates the new pause/resume mechanism in the clarification subgraph
where the system can pause execution, return a question to the chat UI, and resume
when the user responds.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the orchestrator
from src.orchestrator import MainOrchestrator


async def demo_pause_resume_flow():
    """Demo the complete pause/resume clarification flow."""
    logger.info("=" * 80)
    logger.info("Demo: Pause/Resume Clarification Flow")
    logger.info("=" * 80)
    
    # Initialize orchestrator
    orchestrator = MainOrchestrator(use_database=False)
    
    # Step 1: User sends initial request that needs clarification
    logger.info("\n📝 Step 1: User sends initial request")
    logger.info("User: 'Transfer money'")
    
    with patch('src.nodes.intent_classification_node.LLMClient') as mock_intent_llm:
        with patch('src.nodes.missing_param_analysis_node.LLMClient') as mock_missing_llm:
            with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_question_llm:
                
                # Mock intent classification
                mock_intent_llm.return_value.chat_completion.return_value = "action"
                
                # Mock missing parameter analysis
                mock_missing_llm.return_value.chat_completion.return_value = """
                {
                    "extracted_parameters": {"amount": null, "account": null},
                    "missing_params": ["amount", "account"],
                    "missing_critical_params": ["amount"],
                    "parameter_priorities": ["amount", "account"],
                    "normalization_suggestions": {},
                    "ambiguity_flags": {}
                }
                """
                
                # Mock question generation
                mock_question_llm.return_value.chat_completion.return_value = "How much would you like to transfer?"
                
                # Process the initial request
                response1 = await orchestrator.process_user_message(
                    user_input="Transfer money",
                    session_id=str(uuid4())
                )
                
                logger.info(f"System Response: {response1['response']}")
                logger.info(f"Status: {response1['status']}")
                logger.info(f"Session ID: {response1['session_id']}")
                
                # Verify this is a clarification question
                assert response1['status'] == "waiting_for_clarification"
                assert "transfer" in response1['response'].lower()
                
                session_id = response1['session_id']
                
                logger.info("✅ Step 1 completed: System paused and returned clarification question")
    
    # Step 2: User responds to the clarification question
    logger.info("\n📝 Step 2: User responds to clarification question")
    logger.info(f"User: '$500'")
    
    with patch('src.nodes.user_response_processor_node.LLMClient') as mock_response_llm:
        with patch('src.nodes.completeness_validator_node.LLMClient') as mock_validator_llm:
            
            # Mock user response processing
            mock_response_llm.return_value.chat_completion.return_value = """
            {
                "extracted_parameters": {"amount": "$500", "account": null},
                "missing_params": ["account"],
                "missing_critical_params": ["account"],
                "parameter_priorities": ["account"],
                "normalization_suggestions": {},
                "ambiguity_flags": {}
            }
            """
            
            # Mock completeness validation
            mock_validator_llm.return_value.chat_completion.return_value = "incomplete"
            
            # Process the user response
            response2 = await orchestrator.process_user_message(
                user_input="$500",
                session_id=session_id
            )
            
            logger.info(f"System Response: {response2['response']}")
            logger.info(f"Status: {response2['status']}")
            
            # Verify this is another clarification question
            assert response2['status'] == "waiting_for_clarification"
            assert "account" in response2['response'].lower()
            
            logger.info("✅ Step 2 completed: System resumed and asked for more information")
    
    # Step 3: User responds with final information
    logger.info("\n📝 Step 3: User provides final information")
    logger.info(f"User: 'From my checking account'")
    
    with patch('src.nodes.user_response_processor_node.LLMClient') as mock_response_llm:
        with patch('src.nodes.completeness_validator_node.LLMClient') as mock_validator_llm:
            with patch('src.nodes.parameter_extraction_node.LLMClient') as mock_extract_llm:
                with patch('src.nodes.execution_planner_node.LLMClient') as mock_planner_llm:
                    with patch('src.nodes.tool_execution_node.LLMClient') as mock_tool_llm:
                        with patch('src.nodes.response_synthesis_node.LLMClient') as mock_synth_llm:
                            
                            # Mock user response processing (final)
                            mock_response_llm.return_value.chat_completion.return_value = """
                            {
                                "extracted_parameters": {"amount": "$500", "account": "checking"},
                                "missing_params": [],
                                "missing_critical_params": [],
                                "parameter_priorities": [],
                                "normalization_suggestions": {},
                                "ambiguity_flags": {}
                            }
                            """
                            
                            # Mock completeness validation (complete)
                            mock_validator_llm.return_value.chat_completion.return_value = "complete"
                            
                            # Mock parameter extraction
                            mock_extract_llm.return_value.chat_completion.return_value = """
                            {
                                "parameters": {
                                    "amount": 500.0,
                                    "source_account": "checking",
                                    "action_type": "transfer"
                                }
                            }
                            """
                            
                            # Mock execution planning
                            mock_planner_llm.return_value.chat_completion.return_value = """
                            {
                                "plan": [
                                    {"step": "validate_account", "params": {"account": "checking"}},
                                    {"step": "execute_transfer", "params": {"amount": 500.0, "account": "checking"}}
                                ]
                            }
                            """
                            
                            # Mock tool execution
                            mock_tool_llm.return_value.chat_completion.return_value = """
                            {
                                "results": {
                                    "transfer_successful": true,
                                    "transaction_id": "TXN123456"
                                }
                            }
                            """
                            
                            # Mock response synthesis
                            mock_synth_llm.return_value.chat_completion.return_value = "I've successfully transferred $500 from your checking account. Your transaction ID is TXN123456."
                            
                            # Process the final response
                            response3 = await orchestrator.process_user_message(
                                user_input="From my checking account",
                                session_id=session_id
                            )
                            
                            logger.info(f"System Response: {response3['response']}")
                            logger.info(f"Status: {response3['status']}")
                            
                            # Verify this is a final response
                            assert response3['status'] == "success"
                            assert "transferred" in response3['response'].lower()
                            
                            logger.info("✅ Step 3 completed: System completed the transfer successfully")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 Pause/Resume Clarification Flow Demo Completed!")
    logger.info("=" * 80)
    
    # Summary
    logger.info("\n📊 Summary:")
    logger.info("1. User sent initial request → System paused and asked for amount")
    logger.info("2. User provided amount → System resumed and asked for account")
    logger.info("3. User provided account → System completed the transfer")
    logger.info("\n✅ The pause/resume mechanism works correctly!")


async def demo_session_persistence():
    """Demo that session state is properly persisted across pause/resume cycles."""
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Session Persistence Across Pause/Resume")
    logger.info("=" * 80)
    
    orchestrator = MainOrchestrator(use_database=False)
    
    # Create a session and check persistence
    with patch('src.nodes.intent_classification_node.LLMClient') as mock_intent_llm:
        with patch('src.nodes.missing_param_analysis_node.LLMClient') as mock_missing_llm:
            with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_question_llm:
                
                # Mock responses
                mock_intent_llm.return_value.chat_completion.return_value = "action"
                mock_missing_llm.return_value.chat_completion.return_value = """
                {
                    "extracted_parameters": {"amount": null},
                    "missing_params": ["amount"],
                    "missing_critical_params": ["amount"],
                    "parameter_priorities": ["amount"],
                    "normalization_suggestions": {},
                    "ambiguity_flags": {}
                }
                """
                mock_question_llm.return_value.chat_completion.return_value = "How much?"
                
                # First call
                response1 = await orchestrator.process_user_message("Transfer money")
                session_id = response1['session_id']
                
                logger.info(f"Session created: {session_id}")
                logger.info(f"Session stats: {orchestrator.get_orchestrator_stats()}")
                
                # Second call with same session
                response2 = await orchestrator.process_user_message("$100", session_id=session_id)
                
                logger.info(f"Session continued: {response2['session_id']}")
                logger.info(f"Session stats: {orchestrator.get_orchestrator_stats()}")
                
                # Verify same session
                assert response1['session_id'] == response2['session_id']
                
                logger.info("✅ Session persistence works correctly!")


async def main():
    """Run all demos."""
    try:
        await demo_pause_resume_flow()
        await demo_session_persistence()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 All pause/resume demos completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 