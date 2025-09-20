"""
Direct Orchestrator Subgraph for LangGraph Intent Orchestration System.

This subgraph handles the direct execution of user requests when sufficient parameters
are available and confidence is high. It follows a linear flow from parameter extraction
through tool execution to response synthesis.

Flow:
1. Entry State (MainState) → Orchestrator Entry Wrapper → Parameter Extraction
2. Parameter Extraction → Execution Planning → Tool Execution 
3. Tool Execution → Response Synthesis → Orchestrator Exit Wrapper → Exit State (MainState)

Features:
- Automatic state conversion at subgraph boundaries  
- Linear flow with comprehensive error handling
- Tool execution with transaction support
- Response synthesis with LLM integration
- Session and conversation context preservation
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from vyuu_copilot_v2.schemas.state_schemas import (
    MainState,
    OrchestratorState, 
    StateTransitions,
    MessageManager
)
from vyuu_copilot_v2.nodes.parameter_extraction_node import parameter_extraction_node
from vyuu_copilot_v2.nodes.execution_planner_node import execution_planner_node
from vyuu_copilot_v2.nodes.tool_execution_node import tool_execution_node
from vyuu_copilot_v2.nodes.response_synthesis_node import response_synthesis_node

logger = logging.getLogger(__name__)


def reconstruct_state_from_dict(state_dict: Any, target_class: type) -> Any:
    """
    Reconstruct a proper state object from LangGraph's AddableValuesDict or dict.
    
    Args:
        state_dict: State data (could be AddableValuesDict, dict, or state object)
        target_class: Target state class (MainState, OrchestratorState, etc.)
        
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


async def orchestrator_entry_wrapper(state: MainState) -> OrchestratorState:
    """
    Wrapper node to convert MainState to OrchestratorState at subgraph entry.
    
    This is the entry point for the direct orchestrator subgraph. It handles the state
    conversion and adds appropriate tracking messages.
    
    Args:
        state: MainState from decision router or clarification subgraph
        
    Returns:
        OrchestratorState ready for orchestrator processing
    """
    node_name = "orchestrator_entry_wrapper"
    
    try:
        # Ensure we have a proper MainState object
        if not isinstance(state, MainState):
            state = reconstruct_state_from_dict(state, MainState)
        
        logger.info(f"[DirectOrchestrator] Entry wrapper for session {state.session_id[:8]}")
        logger.info(f"[DirectOrchestrator] Entry state user_id: {state.metadata.get('user_id', 'NOT_FOUND')}")
        logger.info(f"[DirectOrchestrator] Entry state metadata keys: {list(state.metadata.keys())}")
        logger.info(f"[DirectOrchestrator] Entry state financial_data present: {'financial_data' in state.metadata}")
        
        # Convert MainState to OrchestratorState
        orchestrator_state = StateTransitions.to_orchestrator_state(state)
        logger.info(f"[DirectOrchestrator] Post-conversion user_id: {orchestrator_state.metadata.get('user_id', 'NOT_FOUND')}")
        logger.info(f"[DirectOrchestrator] Post-conversion metadata keys: {list(orchestrator_state.metadata.keys())}")
        logger.info(f"[DirectOrchestrator] Post-conversion financial_data present: {'financial_data' in orchestrator_state.metadata}")
        
        # Add system message for tracking
        orchestrator_state = MessageManager.add_system_message(
            orchestrator_state,
            f"Entering Direct Orchestrator Subgraph - processing {state.intent} intent with direct execution",
            node_name
        )
        
        # Update metadata to track subgraph entry
        orchestrator_state = orchestrator_state.model_copy(update={
            "metadata": {
                **orchestrator_state.metadata,
                "subgraph": "direct_orchestrator",
                "subgraph_entry_time": "now",
                "orchestrator_status": "initializing",
                "previous_subgraph": orchestrator_state.metadata.get("previous_subgraph")
            }
        })
        
        logger.info(f"[DirectOrchestrator] State conversion successful for session {state.session_id[:8]}")
        return orchestrator_state
        
    except Exception as e:
        logger.error(f"[DirectOrchestrator] Entry wrapper failed: {e}")
        # Create fallback state with error information
        try:
            fallback_state = StateTransitions.to_orchestrator_state(state)
        except Exception:
            # If state conversion fails, create minimal fallback
            fallback_state = OrchestratorState(
                user_input=getattr(state, 'user_input', 'Unknown input'),
                session_id=getattr(state, 'session_id', 'unknown'),
                intent=getattr(state, 'intent', None),
                confidence=getattr(state, 'confidence', 0.0),
                messages=getattr(state, 'messages', []),
                metadata=getattr(state, 'metadata', {}),
                parameters=getattr(state, 'parameters', {})
            )
        
        return fallback_state.model_copy(update={
            "metadata": {
                **fallback_state.metadata,
                "subgraph": "direct_orchestrator",
                "error": f"Entry wrapper failed: {str(e)}",
                "orchestrator_status": "error"
            }
        })


async def orchestrator_exit_wrapper(state: OrchestratorState) -> MainState:
    """
    Wrapper node to convert OrchestratorState back to MainState at subgraph exit.
    
    This handles the state conversion when orchestration is complete and preserves
    the final response and execution results.
    
    Args:
        state: OrchestratorState with completed orchestration process
        
    Returns:
        MainState with final response and execution results
    """
    node_name = "orchestrator_exit_wrapper"
    
    try:
        # Ensure we have a proper OrchestratorState object
        if not isinstance(state, OrchestratorState):
            state = reconstruct_state_from_dict(state, OrchestratorState)
        
        logger.info(f"[DirectOrchestrator] Exit wrapper for session {state.session_id[:8]}")
        
        # Convert OrchestratorState back to MainState
        main_state = StateTransitions.from_orchestrator_state(state)
        
        # Add system message for tracking
        main_state = MessageManager.add_system_message(
            main_state,
            f"Exiting Direct Orchestrator Subgraph - execution complete with final response",
            node_name
        )
        
        # Update metadata to track subgraph exit and completion
        main_state = main_state.model_copy(update={
            "metadata": {
                **main_state.metadata,
                "previous_subgraph": "direct_orchestrator",
                "orchestration_completed": True,
                "final_response_generated": True,
                "tool_execution_results": state.tool_results,
                "execution_plan_used": state.execution_plan,
                "orchestrator_status": "completed"
            }
        })
        
        logger.info(f"[DirectOrchestrator] Exit successful - final response: {main_state.response[:100]}...")
        return main_state
        
    except Exception as e:
        logger.error(f"[DirectOrchestrator] Exit wrapper failed: {e}")
        # Create fallback main state
        try:
            fallback_state = StateTransitions.from_orchestrator_state(state)
        except Exception:
            # If state conversion fails, create minimal fallback
            fallback_state = MainState(
                user_input=getattr(state, 'user_input', 'Unknown input'),
                session_id=getattr(state, 'session_id', 'unknown'),
                intent=getattr(state, 'intent', None),
                confidence=getattr(state, 'confidence', 0.0),
                messages=getattr(state, 'messages', []),
                metadata=getattr(state, 'metadata', {}),
                parameters=getattr(state, 'extracted_params', {}),
                execution_results=getattr(state, 'tool_results', {}),
                response=getattr(state, 'final_response', 'An error occurred while processing your request.')
            )
        
        return fallback_state.model_copy(update={
            "metadata": {
                **fallback_state.metadata,
                "error": f"Exit wrapper failed: {str(e)}",
                "orchestrator_status": "error"
            },
            "response": state.final_response or "An error occurred while processing your request."
        })


def create_direct_orchestrator_subgraph() -> StateGraph:
    """
    Create and compile the direct orchestrator subgraph.
    
    Returns:
        Compiled StateGraph for direct orchestrator processing
    """
    logger.info("[DirectOrchestrator] Creating direct orchestrator subgraph...")
    
    # Create graph with OrchestratorState
    graph = StateGraph(OrchestratorState)
    
    # Add all nodes
    graph.add_node("orchestrator_entry", orchestrator_entry_wrapper)
    graph.add_node("parameter_extraction", parameter_extraction_node)
    graph.add_node("execution_planner", execution_planner_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("response_synthesis", response_synthesis_node)
    graph.add_node("orchestrator_exit", orchestrator_exit_wrapper)
    
    # Define linear flow edges
    graph.add_edge(START, "orchestrator_entry")
    graph.add_edge("orchestrator_entry", "parameter_extraction")
    graph.add_edge("parameter_extraction", "execution_planner")
    graph.add_edge("execution_planner", "tool_execution")
    graph.add_edge("tool_execution", "response_synthesis")
    graph.add_edge("response_synthesis", "orchestrator_exit")
    graph.add_edge("orchestrator_exit", END)
    
    # Compile graph
    try:
        compiled_graph = graph.compile()
        logger.info("[DirectOrchestrator] Direct orchestrator subgraph compiled successfully")
        return compiled_graph
    except Exception as e:
        logger.error(f"[DirectOrchestrator] Failed to compile graph: {e}")
        raise


# Create the direct orchestrator subgraph instance
direct_orchestrator_subgraph = create_direct_orchestrator_subgraph()


# Export for use in main graph
__all__ = [
    "direct_orchestrator_subgraph",
    "create_direct_orchestrator_subgraph",
    "orchestrator_entry_wrapper",
    "orchestrator_exit_wrapper"
] 