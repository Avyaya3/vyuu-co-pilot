"""
Clarification Subgraph for LangGraph Intent Orchestration System.

This subgraph handles parameter collection through clarification questions when the
decision router determines that additional information is needed from the user.

Flow:
1. Entry State (MainState) → Clarification Entry Wrapper → Missing Parameter Analysis
2. Missing Parameter Analysis → Clarification Question Generator → [PAUSE & RETURN]
3. User Response → Clarification Resume Node → User Response Processor → Completeness Validator
4. Completeness Validator:
   - "complete" → Exit to Direct Orchestrator (via wrapper)
   - "incomplete" → Loop back to Missing Parameter Analysis (if attempts < max)
   - "max_attempts_reached" → Exit with Partial Data

Features:
- Automatic state conversion at subgraph boundaries
- Conditional routing based on validation results
- Maximum attempt handling with graceful exit
- Comprehensive error handling and recovery
- Session and conversation context preservation
- Pause/resume mechanism for real user interaction
"""

import logging
from typing import Dict, Any, Literal, Union
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.schemas.state_schemas import (
    MainState, 
    ClarificationState, 
    StateTransitions,
    MessageManager
)
from src.nodes.missing_param_analysis_node import missing_param_analysis_node
from src.nodes.clarification_question_generator_node import clarification_question_generator_node
from src.nodes.clarification_resume_node import clarification_resume_node
from src.nodes.user_response_processor_node import user_response_processor_node
from src.nodes.completeness_validator_node import completeness_validator_node
from src.nodes.exit_with_partial_data_node import exit_with_partial_data_node

logger = logging.getLogger(__name__)

# Type definitions for routing
ClarificationRoutingDecision = Literal["continue_clarification", "exit_to_orchestrator", "exit_with_partial_data", "pause_for_user_response", "resume_processing"]


def reconstruct_state_from_dict(state_dict: Any, target_class: type) -> Any:
    """
    Reconstruct a proper state object from LangGraph's AddableValuesDict or dict.
    
    Args:
        state_dict: State data (could be AddableValuesDict, dict, or state object)
        target_class: Target state class (MainState, ClarificationState, etc.)
        
    Returns:
        Properly constructed state object
    """
    try:
        # If it's already the right type, return as is
        if isinstance(state_dict, target_class):
            return state_dict
        
        # Convert to dict if it's an AddableValuesDict or similar
        if hasattr(state_dict, '__dict__'):
            # Handle AddableValuesDict and similar objects
            state_data = dict(state_dict)
        elif isinstance(state_dict, dict):
            state_data = state_dict
        else:
            # Try to convert to dict using model_dump if it's a Pydantic model
            try:
                state_data = state_dict.model_dump()
            except AttributeError:
                logger.error(f"Cannot convert state to dict: {type(state_dict)}")
                raise ValueError(f"Unsupported state type: {type(state_dict)}")
        
        # Create new instance of target class
        return target_class(**state_data)
        
    except Exception as e:
        logger.error(f"Failed to reconstruct state: {e}")
        raise


async def clarification_entry_wrapper(state: MainState) -> ClarificationState:
    """
    Wrapper node to convert MainState to ClarificationState at subgraph entry.
    
    This is the entry point for the clarification subgraph. It handles the state
    conversion and adds appropriate tracking messages.
    
    Args:
        state: MainState from decision router with routing decision
        
    Returns:
        ClarificationState ready for clarification processing
    """
    node_name = "clarification_entry_wrapper"
    
    try:
        # Ensure we have a proper MainState object
        if not isinstance(state, MainState):
            state = reconstruct_state_from_dict(state, MainState)
        
        logger.info(f"[ClarificationSubgraph] Entry wrapper for session {state.session_id[:8]}")
        
        # Convert MainState to ClarificationState
        clarification_state = StateTransitions.to_clarification_state(state)
        
        # Add system message for tracking
        clarification_state = MessageManager.add_system_message(
            clarification_state,
            f"Entering Clarification Subgraph - collecting additional parameters for {state.intent} intent",
            node_name
        )
        
        # Update metadata to track subgraph entry
        clarification_state = clarification_state.model_copy(update={
            "metadata": {
                **clarification_state.metadata,
                "subgraph": "clarification",
                "subgraph_entry_time": "now",
                "clarification_status": "initializing"
            }
        })
        
        logger.info(f"[ClarificationSubgraph] State conversion successful for session {state.session_id[:8]}")
        return clarification_state
        
    except Exception as e:
        logger.error(f"[ClarificationSubgraph] Entry wrapper failed: {e}")
        # Create fallback state with error information
        try:
            fallback_state = StateTransitions.to_clarification_state(state)
        except Exception:
            # If state conversion fails, create minimal fallback
            fallback_state = ClarificationState(
                user_input=getattr(state, 'user_input', 'Unknown input'),
                session_id=getattr(state, 'session_id', 'unknown'),
                intent=getattr(state, 'intent', None),
                confidence=getattr(state, 'confidence', 0.0),
                messages=getattr(state, 'messages', []),
                metadata=getattr(state, 'metadata', {})
            )
        
        return fallback_state.model_copy(update={
            "metadata": {
                **fallback_state.metadata,
                "subgraph": "clarification",
                "error": f"Entry wrapper failed: {str(e)}",
                "clarification_status": "error"
            }
        })


async def clarification_exit_wrapper(state: ClarificationState) -> MainState:
    """
    Wrapper node to convert ClarificationState back to MainState at subgraph exit.
    
    This handles the state conversion when clarification is complete and merges
    the clarified parameters into the main state.
    
    Args:
        state: ClarificationState with completed clarification process
        
    Returns:
        MainState with merged clarified parameters
    """
    node_name = "clarification_exit_wrapper"
    
    try:
        # Ensure we have a proper ClarificationState object
        if not isinstance(state, ClarificationState):
            state = reconstruct_state_from_dict(state, ClarificationState)
        
        logger.info(f"[ClarificationSubgraph] Exit wrapper for session {state.session_id[:8]}")
        
        # Convert ClarificationState back to MainState (automatic parameter merging)
        main_state = StateTransitions.from_clarification_state(state)
        
        # Add system message for tracking
        main_state = MessageManager.add_system_message(
            main_state,
            f"Exiting Clarification Subgraph - parameters collected, routing to Direct Orchestrator",
            node_name
        )
        
        # Update metadata to track subgraph exit
        main_state = main_state.model_copy(update={
            "metadata": {
                **main_state.metadata,
                "previous_subgraph": "clarification",
                "clarification_completed": True,
                "routing_decision": "direct_orchestrator",  # Set routing for main graph
                "clarified_parameters": state.extracted_parameters,
                "clarification_attempts_used": state.clarification_attempts
            }
        })
        
        logger.info(f"[ClarificationSubgraph] Exit successful - {len(state.extracted_parameters)} parameters collected")
        return main_state
        
    except Exception as e:
        logger.error(f"[ClarificationSubgraph] Exit wrapper failed: {e}")
        # Create fallback main state
        try:
            fallback_state = StateTransitions.from_clarification_state(state)
        except Exception:
            # If state conversion fails, create minimal fallback
            fallback_state = MainState(
                user_input=getattr(state, 'user_input', 'Unknown input'),
                session_id=getattr(state, 'session_id', 'unknown'),
                intent=getattr(state, 'intent', None),
                confidence=getattr(state, 'confidence', 0.0),
                messages=getattr(state, 'messages', []),
                metadata=getattr(state, 'metadata', {}),
                parameters=getattr(state, 'extracted_parameters', {})
            )
        
        return fallback_state.model_copy(update={
            "metadata": {
                **fallback_state.metadata,
                "error": f"Exit wrapper failed: {str(e)}",
                "routing_decision": "direct_orchestrator"  # Continue despite error
            }
        })


def get_clarification_routing_decision(state: ClarificationState) -> ClarificationRoutingDecision:
    """
    Determine routing decision based on clarification state and validation results.
    
    This function is used by LangGraph conditional edges to route based on the
    current clarification phase and validation results.
    
    Args:
        state: ClarificationState with current phase and validation results
        
    Returns:
        Routing decision for conditional edges
    """
    try:
        # Ensure we have a proper ClarificationState object
        if not isinstance(state, ClarificationState):
            state = reconstruct_state_from_dict(state, ClarificationState)
        
        # Check if we're waiting for user response (pause state)
        if state.waiting_for_response and state.pending_question:
            logger.info(f"[ClarificationRouting] Paused waiting for user response for session {state.session_id[:8]}")
            return "pause_for_user_response"
        
        # Check if we're resuming from a pause
        if state.clarification_phase == "processing" and state.metadata.get("resumed_from_pause"):
            logger.info(f"[ClarificationRouting] Resuming processing for session {state.session_id[:8]}")
            return "resume_processing"
        
        # Check for explicit exit signals
        clarification_status = state.metadata.get("clarification_status")
        
        if clarification_status == "complete":
            logger.info(f"[ClarificationRouting] Complete - routing to orchestrator for session {state.session_id[:8]}")
            return "exit_to_orchestrator"
        
        elif clarification_status == "max_attempts_reached":
            logger.info(f"[ClarificationRouting] Max attempts reached - exiting with partial data for session {state.session_id[:8]}")
            return "exit_with_partial_data"
        
        elif clarification_status == "incomplete":
            if state.clarification_attempts >= state.max_attempts:
                logger.warning(f"[ClarificationRouting] Attempts exhausted ({state.clarification_attempts}/{state.max_attempts}) for session {state.session_id[:8]}")
                return "exit_with_partial_data"
            else:
                logger.info(f"[ClarificationRouting] Continue clarification ({state.clarification_attempts}/{state.max_attempts}) for session {state.session_id[:8]}")
                return "continue_clarification"
        
        else:
            # Default case - continue clarification
            logger.warning(f"[ClarificationRouting] Unknown status '{clarification_status}' - defaulting to continue for session {state.session_id[:8]}")
            return "continue_clarification"
            
    except Exception as e:
        logger.error(f"[ClarificationRouting] Error determining route: {e}")
        return "continue_clarification"  # Safe default


def create_clarification_subgraph() -> StateGraph:
    """
    Create and compile the clarification subgraph.
    
    Returns:
        Compiled StateGraph for clarification processing
    """
    logger.info("[ClarificationSubgraph] Creating clarification subgraph...")
    
    # Create graph with ClarificationState
    graph = StateGraph(ClarificationState)
    
    # Add all nodes
    graph.add_node("clarification_entry", clarification_entry_wrapper)
    graph.add_node("missing_param_analysis", missing_param_analysis_node)
    graph.add_node("clarification_question_generator", clarification_question_generator_node)
    graph.add_node("clarification_resume", clarification_resume_node)
    graph.add_node("user_response_processor", user_response_processor_node)
    graph.add_node("completeness_validator", completeness_validator_node)
    graph.add_node("clarification_exit", clarification_exit_wrapper)
    graph.add_node("exit_with_partial_data", exit_with_partial_data_node)
    
    # Define edges
    # Entry flow
    graph.add_edge(START, "clarification_entry")
    graph.add_edge("clarification_entry", "missing_param_analysis")
    
    # Main clarification loop with pause/resume
    graph.add_edge("missing_param_analysis", "clarification_question_generator")
    
    # Conditional routing from question generator
    graph.add_conditional_edges(
        "clarification_question_generator",
        get_clarification_routing_decision,
        {
            "pause_for_user_response": "clarification_pause",  # Pause and return to main graph
            "continue_clarification": "user_response_processor",  # Direct processing (fallback)
            "exit_to_orchestrator": "clarification_exit",
            "exit_with_partial_data": "exit_with_partial_data"
        }
    )
    
    # Resume flow (when user responds)
    graph.add_edge("clarification_resume", "user_response_processor")
    
    # Continue flow from user response processor
    graph.add_edge("user_response_processor", "completeness_validator")
    
    # Conditional routing from completeness validator
    graph.add_conditional_edges(
        "completeness_validator",
        get_clarification_routing_decision,
        {
            "continue_clarification": "missing_param_analysis",  # Loop back
            "resume_processing": "missing_param_analysis",       # Resume processing
            "exit_to_orchestrator": "clarification_exit",        # Complete
            "exit_with_partial_data": "exit_with_partial_data"   # Max attempts
        }
    )
    
    # Exit edges
    graph.add_edge("clarification_exit", END)
    graph.add_edge("exit_with_partial_data", END)
    
    # Pause node (returns to main graph)
    graph.add_node("clarification_pause", lambda state: state)  # Identity function to pause
    graph.add_edge("clarification_pause", END)
    
    # Compile graph
    try:
        compiled_graph = graph.compile()
        logger.info("[ClarificationSubgraph] Clarification subgraph compiled successfully")
        return compiled_graph
    except Exception as e:
        logger.error(f"[ClarificationSubgraph] Failed to compile graph: {e}")
        raise


# Create the clarification subgraph instance
clarification_subgraph = create_clarification_subgraph()


# Export for use in main graph
__all__ = [
    "clarification_subgraph",
    "create_clarification_subgraph",
    "clarification_entry_wrapper",
    "clarification_exit_wrapper",
    "get_clarification_routing_decision"
] 