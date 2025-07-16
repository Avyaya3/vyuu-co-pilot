import logging
from typing import Tuple
from src.schemas.state_schemas import ClarificationState

logger = logging.getLogger(__name__)


def exit_with_partial_data_node(
    state: ClarificationState,
) -> Tuple[str, ClarificationState]:
    """
    Handle graceful exit from clarification subgraph when max attempts are reached.

    This node is the final step in the clarification subgraph flow when the system
    decides to exit with partial data after reaching maximum clarification attempts.

    Expected Flow:
    1. ClarificationQuestionGenerator reaches max attempts
    2. Returns "EXIT_WITH_PARTIAL_DATA" signal
    3. Graph routes to this exit handler node
    4. This node delivers the pre-crafted exit message and closes subgraph

    Args:
        state: ClarificationState with metadata indicating max attempts reached

    Returns:
        Tuple of (exit_message, updated_state) where:
        - exit_message: User-friendly message from state.metadata["exit_message"]
        - updated_state: State marked with clarification_subgraph_closed = True

    Raises:
        AssertionError: If state is not in expected max_attempts_reached status
        KeyError: If required exit_message is missing from metadata
    """
    logger.info(
        f"[ExitWithPartialData] Processing exit for session {state.session_id[:8]}"
    )

    # Assert that we're in the expected state
    clarification_status = state.metadata.get("clarification_status")
    assert (
        clarification_status == "max_attempts_reached"
    ), f"Expected clarification_status 'max_attempts_reached', got '{clarification_status}'"

    # Retrieve the pre-crafted exit message
    try:
        exit_message = state.metadata["exit_message"]
    except KeyError:
        logger.error(
            f"[ExitWithPartialData] Missing exit_message in metadata for session {state.session_id[:8]}"
        )
        raise KeyError("Required 'exit_message' not found in state.metadata")

    # Log the exit
    logger.info(f"[ExitWithPartialData] Exiting clarification subgraph: {exit_message}")

    # Update state to mark subgraph as closed
    updated_state = state.model_copy(
        update={"metadata": {**state.metadata, "clarification_subgraph_closed": True}}
    )

    logger.info(
        f"[ExitWithPartialData] Subgraph closed for session {state.session_id[:8]}"
    )

    return exit_message, updated_state
