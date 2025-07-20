"""
Main Orchestrator Graph for LangGraph Intent Orchestration System.

This is the complete graph assembly that combines all nodes and subgraphs into
the final intent orchestration workflow. It handles the full flow from user
input through intent classification, decision routing, and execution.

Main Flow:
User Input → Intent Classification → Decision Router
                                    ↓
          ┌─────────────────────────┴─────────────────────┐
          ↓                                           ↓
  Clarification Subgraph                    Direct Orchestrator Subgraph
          ↓                                           ↓
    Final Response                              Final Response

Features:
- Complete intent orchestration workflow
- Conditional routing between subgraphs
- Comprehensive error handling and recovery
- Session and conversation context preservation
- Structured logging and monitoring
"""

import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.schemas.state_schemas import MainState, MessageManager
from src.nodes.intent_classification_node import intent_classification_node
from src.nodes.decision_router_node import decision_router_node, get_routing_decision
from src.subgraphs.clarification_subgraph import clarification_subgraph
from src.subgraphs.direct_orchestrator_subgraph import direct_orchestrator_subgraph

logger = logging.getLogger(__name__)

# Type definitions for main graph routing
MainGraphRoutingDecision = Literal["clarification", "direct_orchestrator"]


async def user_input_processor(state: MainState) -> MainState:
    """
    Process initial user input and prepare state for intent classification.
    
    This is the entry point for the main graph. It handles user input validation,
    session initialization, and conversation context setup.
    
    Args:
        state: MainState with user input
        
    Returns:
        MainState ready for intent classification
    """
    node_name = "user_input_processor"
    
    try:
        logger.info(f"[MainGraph] Processing user input for session {state.session_id[:8]}")
        
        # Add user message to conversation history
        state = MessageManager.add_user_message(state, state.user_input)
        
        # Add system message for tracking
        state = MessageManager.add_system_message(
            state,
            f"Starting intent orchestration for user input: '{state.user_input[:50]}...'",
            node_name
        )
        
        # Update metadata to track main graph entry
        state = state.model_copy(update={
            "metadata": {
                **state.metadata,
                "main_graph_status": "processing",
                "main_graph_entry_time": "now",
                "user_input_processed": True
            }
        })
        
        logger.info(f"[MainGraph] User input processed successfully for session {state.session_id[:8]}")
        return state
        
    except Exception as e:
        logger.error(f"[MainGraph] User input processing failed: {e}")
        # Add error message and continue
        state = MessageManager.add_system_message(
            state,
            f"Error processing user input: {str(e)}",
            node_name
        )
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "main_graph_status": "error",
                "error": f"User input processing failed: {str(e)}"
            }
        })


async def final_response_formatter(state: MainState) -> MainState:
    """
    Format and finalize the response before returning to the user.
    
    This is the final node in the main graph that ensures the response is
    properly formatted and conversation history is updated.
    
    Args:
        state: MainState with response from subgraphs
        
    Returns:
        MainState with finalized response
    """
    node_name = "final_response_formatter"
    
    try:
        logger.info(f"[MainGraph] Formatting final response for session {state.session_id[:8]}")
        
        # Ensure response is set
        if not state.response:
            fallback_response = "I'm sorry, I wasn't able to process your request completely. Please try again."
            state = state.model_copy(update={"response": fallback_response})
            logger.warning(f"[MainGraph] No response found, using fallback for session {state.session_id[:8]}")
        
        # Add assistant response to conversation history
        state = MessageManager.add_assistant_message(
            state,
            state.response,
            node_name
        )
        
        # Update metadata to track completion
        state = state.model_copy(update={
            "metadata": {
                **state.metadata,
                "main_graph_status": "completed",
                "main_graph_completion_time": "now",
                "final_response_formatted": True,
                "conversation_turn_completed": True
            }
        })
        
        logger.info(f"[MainGraph] Final response formatted successfully for session {state.session_id[:8]}")
        return state
        
    except Exception as e:
        logger.error(f"[MainGraph] Final response formatting failed: {e}")
        # Ensure we still return a response
        fallback_response = "An error occurred while formatting the response."
        return state.model_copy(update={
            "response": fallback_response,
            "metadata": {
                **state.metadata,
                "main_graph_status": "error",
                "error": f"Final response formatting failed: {str(e)}"
            }
        })


def get_main_graph_routing_decision(state: MainState) -> MainGraphRoutingDecision:
    """
    Extract routing decision from state metadata for main graph conditional routing.
    
    This function is used by LangGraph to determine which subgraph to route to
    based on the decision router's output or resume state.
    
    Args:
        state: MainState with routing decision in metadata
        
    Returns:
        Routing decision for main graph conditional edges
    """
    try:
        # Check if this is a clarification resume scenario
        if hasattr(state, 'clarification_phase') and state.clarification_phase == "processing":
            logger.info(f"[MainGraphRouting] Resuming clarification subgraph for session {state.session_id[:8]}")
            return "clarification"
        
        # Check if we have a pending clarification question
        if hasattr(state, 'waiting_for_response') and state.waiting_for_response:
            logger.info(f"[MainGraphRouting] Already in clarification subgraph for session {state.session_id[:8]}")
            return "clarification"
        
        # Check metadata for routing decision
        routing_decision = state.metadata.get("routing_decision")
        
        if routing_decision == "clarification":
            logger.info(f"[MainGraphRouting] Routing to clarification subgraph for session {state.session_id[:8]}")
            return "clarification"
        elif routing_decision == "direct_orchestrator":
            logger.info(f"[MainGraphRouting] Routing to direct orchestrator subgraph for session {state.session_id[:8]}")
            return "direct_orchestrator"
        else:
            # Default to clarification for safety
            logger.warning(f"[MainGraphRouting] Unknown routing decision '{routing_decision}', defaulting to clarification for session {state.session_id[:8]}")
            return "clarification"
            
    except Exception as e:
        logger.error(f"[MainGraphRouting] Error extracting routing decision: {e}")
        return "clarification"  # Safe default


def create_main_orchestrator_graph() -> StateGraph:
    """
    Create and compile the complete main orchestrator graph.
    
    Returns:
        Compiled StateGraph for the complete intent orchestration system
    """
    logger.info("[MainGraph] Creating main orchestrator graph...")
    
    # Create graph with MainState
    graph = StateGraph(MainState)
    
    # Add core nodes
    graph.add_node("user_input_processor", user_input_processor)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("decision_router", decision_router_node)
    graph.add_node("final_response_formatter", final_response_formatter)
    
    # Add subgraphs as nodes
    graph.add_node("clarification_subgraph", clarification_subgraph)
    graph.add_node("direct_orchestrator_subgraph", direct_orchestrator_subgraph)
    
    # Define main flow edges
    graph.add_edge(START, "user_input_processor")
    graph.add_edge("user_input_processor", "intent_classification")
    graph.add_edge("intent_classification", "decision_router")
    
    # Conditional routing from decision router to subgraphs
    graph.add_conditional_edges(
        "decision_router",
        get_main_graph_routing_decision,
        {
            "clarification": "clarification_subgraph",
            "direct_orchestrator": "direct_orchestrator_subgraph"
        }
    )
    
    # Both subgraphs route to final response formatter
    graph.add_edge("clarification_subgraph", "final_response_formatter")
    graph.add_edge("direct_orchestrator_subgraph", "final_response_formatter")
    
    # Final edge to END
    graph.add_edge("final_response_formatter", END)
    
    # Compile graph
    try:
        compiled_graph = graph.compile()
        logger.info("[MainGraph] Main orchestrator graph compiled successfully")
        return compiled_graph
    except Exception as e:
        logger.error(f"[MainGraph] Failed to compile main graph: {e}")
        raise


# Create the main orchestrator graph instance
main_orchestrator_graph = create_main_orchestrator_graph()


# Export for use by orchestrator
__all__ = [
    "main_orchestrator_graph",
    "create_main_orchestrator_graph",
    "user_input_processor",
    "final_response_formatter",
    "get_main_graph_routing_decision"
] 