"""
LangGraph State Schema Definitions with Hierarchical Inheritance.

This module provides comprehensive state management schemas for LangGraph nodes and subgraphs,
featuring conversation tracking, automatic state transitions, and message management utilities.

Features:
- Hierarchical inheritance pattern (BaseState -> MainState -> specialized states)
- Automatic conversation pruning (max 20 messages)
- Node tracking in message metadata for debugging
- State transition utilities with parameter merging
- Comprehensive validation with custom validators
- Conversation context and summarization utilities
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, validator


# Configuration Constants
MAX_MESSAGE_LENGTH = 10000
MAX_CONVERSATION_HISTORY = 20
MAX_CLARIFICATION_ATTEMPTS = 3


# Enums
class MessageRole(str, Enum):
    """Message role types for conversation tracking."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class IntentType(str, Enum):
    """Supported intent types for user requests."""
    READ = "read"
    DATABASE_OPERATIONS = "database_operations"
    ADVICE = "advice"
    CLARIFICATION = "clarification"
    UNKNOWN = "unknown"


# Message Schema
class Message(BaseModel):
    """
    Individual message in conversation with role, content, and metadata tracking.
    
    Features:
    - Role validation (user/assistant/system)
    - Content length limits
    - Node tracking in metadata
    - Automatic timestamping
    """
    
    role: MessageRole = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=MAX_MESSAGE_LENGTH,
        description="Message content with length validation"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp (UTC)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Message metadata including node_name for tracking"
    )
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate message content is not empty after stripping."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    @property
    def node_name(self) -> Optional[str]:
        """Get the node name that generated this message."""
        return self.metadata.get('node_name')
    
    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role == MessageRole.USER
    
    @property
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == MessageRole.ASSISTANT
    
    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message."""
        return self.role == MessageRole.SYSTEM


# Base State Schema
class BaseState(BaseModel):
    """
    Base state schema with core conversation and intent tracking.
    
    Features:
    - User input validation
    - Intent and confidence tracking
    - Automatic conversation history management
    - Session and timestamp tracking
    - Extensible metadata
    """
    
    user_input: str = Field(
        ..., 
        min_length=1,
        description="Current user input with non-empty validation"
    )
    intent: Optional[IntentType] = Field(
        None,
        description="Classified intent type"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Intent classification confidence (0.0-1.0)"
    )
    messages: List[Message] = Field(
        default_factory=list,
        description="Conversation history with automatic pruning"
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session identifier (UUID format)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State creation timestamp (UTC)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Extensible metadata for additional context"
    )
    
    @field_validator('user_input')
    @classmethod
    def validate_user_input(cls, v):
        """Validate user input is not empty after stripping."""
        if not v.strip():
            raise ValueError("User input cannot be empty")
        return v.strip()
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id is a valid UUID format."""
        try:
            UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID format")
        return v
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        """Ensure conversation history doesn't exceed maximum length."""
        if len(v) > MAX_CONVERSATION_HISTORY:
            # Keep only the most recent messages
            return v[-MAX_CONVERSATION_HISTORY:]
        return v


# Main State Schema
class MainState(BaseState):
    """
    Main state schema extending BaseState with parameter and execution tracking.
    
    Features:
    - Flexible parameter structure for different intent types
    - Execution results tracking
    - Final response management
    - Multi-intent support with backward compatibility
    """
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted/clarified parameters for execution"
    )
    execution_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Tool execution outcomes and results"
    )
    response: Optional[str] = Field(
        None,
        description="Final user-facing response"
    )
    multiple_intents: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of classified intents when multiple intents detected"
    )
    
    @property
    def has_multiple_intents(self) -> bool:
        """Check if this state contains multiple intents."""
        return self.multiple_intents is not None and len(self.multiple_intents) > 1
    
    @property
    def primary_intent(self) -> Optional[IntentType]:
        """Get primary intent (highest confidence) for backward compatibility."""
        if self.has_multiple_intents:
            sorted_intents = sorted(
                self.multiple_intents, 
                key=lambda x: x.get('confidence', 0.0), 
                reverse=True
            )
            # Convert string intent to IntentType enum
            intent_mapping = {
                "read": IntentType.READ,
                "database_operations": IntentType.DATABASE_OPERATIONS,
                "advice": IntentType.ADVICE,
                "unknown": IntentType.UNKNOWN,
                "clarification": IntentType.CLARIFICATION,
            }
            return intent_mapping.get(sorted_intents[0].get('intent', 'unknown'), IntentType.UNKNOWN)
        return self.intent
    
    @property
    def primary_confidence(self) -> Optional[float]:
        """Get primary intent confidence for backward compatibility."""
        if self.has_multiple_intents:
            sorted_intents = sorted(
                self.multiple_intents, 
                key=lambda x: x.get('confidence', 0.0), 
                reverse=True
            )
            return sorted_intents[0].get('confidence', 0.0)
        return self.confidence


# Clarification State Schema
class ClarificationState(MainState):
    """
    Clarification state schema for parameter collection and validation.
    
    Features:
    - Missing parameter tracking (critical and non-critical)
    - Clarification attempt counting with limits
    - Parameter prioritization and normalization
    - Ambiguity flagging
    - Clarification turn history
    - Automatic parameter merging
    - Pause/resume mechanism for user interaction
    """
    
    missing_params: list[str] = Field(
        default_factory=list,
        description="Required parameters not yet collected"
    )
    missing_critical_params: list[str] = Field(
        default_factory=list,
        description="Subset of missing_params that are critical slots"
    )
    parameter_priorities: list[str] = Field(
        default_factory=list,
        description="Missing parameters ordered by question‑asking priority"
    )
    normalization_suggestions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of user synonyms to canonical slot values suggested by LLM"
    )
    ambiguity_flags: dict[str, str] = Field(
        default_factory=dict,
        description="Any slots whose values are ambiguous and need disambiguation"
    )
    clarification_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Record of past clarification turns (questions asked and user answers)"
    )
    clarification_attempts: int = Field(
        0,
        ge=0,
        description="Number of clarification attempts made"
    )
    max_attempts: int = Field(
        MAX_CLARIFICATION_ATTEMPTS,
        ge=1,
        description="Maximum allowed clarification attempts"
    )
    extracted_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters extracted and normalized by the Missing‑Parameter Analysis node"
    )
    
    # New fields for pause/resume mechanism
    pending_question: Optional[str] = Field(
        None,
        description="Question waiting to be asked to user"
    )
    waiting_for_response: bool = Field(
        False,
        description="True if subgraph is paused waiting for user response"
    )
    clarification_phase: Literal["generating", "waiting", "processing"] = Field(
        "generating",
        description="Current phase of clarification flow"
    )
    last_question_asked: Optional[str] = Field(
        None,
        description="Last question asked to user for context"
    )

    @model_validator(mode='after')
    def validate_clarification_attempts(self):
        if self.clarification_attempts > self.max_attempts:
            raise ValueError(f"Clarification attempts ({self.clarification_attempts}) cannot exceed max_attempts ({self.max_attempts})")
        return self

    @property
    def can_attempt_clarification(self) -> bool:
        """Check if more clarification attempts are allowed."""
        return self.clarification_attempts < self.max_attempts

    @property
    def has_missing_params(self) -> bool:
        """Check if there are still missing parameters."""
        return len(self.missing_params) > 0


# Orchestrator State Schema
class OrchestratorState(MainState):
    """
    Orchestrator state schema for tool execution and response synthesis.
    
    Features:
    - Validated parameter tracking
    - Execution plan management
    - Individual tool result tracking
    - Natural language response synthesis
    """
    
    extracted_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validated parameters ready for execution"
    )
    execution_plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Tool sequence and parameters for execution"
    )
    tool_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Individual tool execution results"
    )
    final_response: Optional[str] = Field(
        None,
        description="Synthesized natural language response"
    )


# State Management Utilities
class StateTransitions:
    """Utility class for automatic state transitions and parameter merging."""
    
    @staticmethod
    def to_clarification_state(main_state: MainState) -> ClarificationState:
        """
        Convert MainState to ClarificationState preserving all fields.
        
        Args:
            main_state: Source MainState to convert
            
        Returns:
            ClarificationState with inherited fields
        """
        return ClarificationState(
            **main_state.model_dump(),
            missing_params=[],
            clarification_attempts=0,
            extracted_parameters={}
        )
    
    @staticmethod
    def from_clarification_state(clarification_state: ClarificationState) -> MainState:
        """
        Convert ClarificationState back to MainState merging extracted parameters.
        
        Args:
            clarification_state: Source ClarificationState to convert
            
        Returns:
            MainState with merged parameters from clarification
        """
        # Merge extracted_parameters into parameters
        merged_params = {**clarification_state.parameters, **clarification_state.extracted_parameters}
        
        return MainState(
            user_input=clarification_state.user_input,
            intent=clarification_state.intent,
            confidence=clarification_state.confidence,
            messages=clarification_state.messages,
            session_id=clarification_state.session_id,
            timestamp=clarification_state.timestamp,
            metadata=clarification_state.metadata,
            parameters=merged_params,
            execution_results=clarification_state.execution_results,
            response=clarification_state.response
        )
    
    @staticmethod
    def to_orchestrator_state(main_state: MainState) -> OrchestratorState:
        """
        Convert MainState to OrchestratorState preserving all fields.
        
        Args:
            main_state: Source MainState to convert
            
        Returns:
            OrchestratorState with inherited fields
        """
        return OrchestratorState(
            **main_state.model_dump(),
            extracted_params=main_state.parameters.copy(),
            execution_plan=None,
            tool_results=None,
            final_response=None
        )
    
    @staticmethod
    def from_orchestrator_state(orchestrator_state: OrchestratorState) -> MainState:
        """
        Convert OrchestratorState back to MainState merging execution results.
        
        Args:
            orchestrator_state: Source OrchestratorState to convert
            
        Returns:
            MainState with merged execution results
        """
        return MainState(
            user_input=orchestrator_state.user_input,
            intent=orchestrator_state.intent,
            confidence=orchestrator_state.confidence,
            messages=orchestrator_state.messages,
            session_id=orchestrator_state.session_id,
            timestamp=orchestrator_state.timestamp,
            metadata=orchestrator_state.metadata,
            parameters=orchestrator_state.extracted_params,
            execution_results=orchestrator_state.tool_results,
            response=orchestrator_state.final_response
        )


class MessageManager:
    """Utility class for message management and conversation handling."""
    
    @staticmethod
    def add_user_message(state: BaseState, content: str) -> BaseState:
        """
        Add user message to conversation with automatic timestamp.
        
        Args:
            state: Current state to update
            content: User message content
            
        Returns:
            Updated state with new user message
        """
        message = Message(
            role=MessageRole.USER,
            content=content,
            metadata={"source": "user_input"}
        )
        
        new_messages = state.messages + [message]
        # Automatic pruning
        if len(new_messages) > MAX_CONVERSATION_HISTORY:
            new_messages = new_messages[-MAX_CONVERSATION_HISTORY:]
        
        state.messages = new_messages
        return state
    
    @staticmethod
    def add_assistant_message(state: BaseState, content: str, node_name: str) -> BaseState:
        """
        Add assistant message with node tracking.
        
        Args:
            state: Current state to update
            content: Assistant message content
            node_name: Name of the node generating the message
            
        Returns:
            Updated state with new assistant message
        """
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "node_name": node_name,
                "source": "assistant_response"
            }
        )
        
        new_messages = state.messages + [message]
        # Automatic pruning
        if len(new_messages) > MAX_CONVERSATION_HISTORY:
            new_messages = new_messages[-MAX_CONVERSATION_HISTORY:]
        
        state.messages = new_messages
        return state
    
    @staticmethod
    def add_system_message(state: BaseState, content: str, node_name: str) -> BaseState:
        """
        Add system message for debugging and tracking.
        
        Args:
            state: Current state to update
            content: System message content
            node_name: Name of the node generating the message
            
        Returns:
            Updated state with new system message
        """
        message = Message(
            role=MessageRole.SYSTEM,
            content=content,
            metadata={
                "node_name": node_name,
                "source": "system_debug"
            }
        )
        
        new_messages = state.messages + [message]
        # Automatic pruning
        if len(new_messages) > MAX_CONVERSATION_HISTORY:
            new_messages = new_messages[-MAX_CONVERSATION_HISTORY:]
        
        state.messages = new_messages
        return state
    
    @staticmethod
    def prune_conversation(state: BaseState, max_messages: int = MAX_CONVERSATION_HISTORY) -> BaseState:
        """
        Manually prune conversation history to specified limit.
        
        Args:
            state: Current state to update
            max_messages: Maximum number of messages to keep
            
        Returns:
            Updated state with pruned conversation
        """
        if len(state.messages) > max_messages:
            state.messages = state.messages[-max_messages:]
        return state


class ConversationContext:
    """Utility class for conversation context and summarization."""
    
    @staticmethod
    def get_recent_context(state: BaseState, num_messages: int = 5) -> List[Message]:
        """
        Extract recent relevant messages for context.
        
        Args:
            state: Current state
            num_messages: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return state.messages[-num_messages:] if state.messages else []
    
    @staticmethod
    def summarize_conversation(state: BaseState) -> str:
        """
        Create conversation summary for long chats.
        
        Args:
            state: Current state with conversation history
            
        Returns:
            String summary of conversation
        """
        if not state.messages:
            return "No conversation history"
        
        user_messages = [msg for msg in state.messages if msg.is_user_message]
        assistant_messages = [msg for msg in state.messages if msg.is_assistant_message]
        system_messages = [msg for msg in state.messages if msg.is_system_message]
        
        summary = f"Conversation Summary (Session: {state.session_id[:8]}...):\n"
        summary += f"- Total messages: {len(state.messages)}\n"
        summary += f"- User messages: {len(user_messages)}\n"
        summary += f"- Assistant messages: {len(assistant_messages)}\n"
        summary += f"- System messages: {len(system_messages)}\n"
        summary += f"- Current intent: {state.intent}\n"
        summary += f"- Intent confidence: {state.confidence}\n"
        
        if state.messages:
            summary += f"- Latest message: {state.messages[-1].role} - {state.messages[-1].content[:50]}..."
        
        return summary
    
    @staticmethod
    def get_messages_by_role(state: BaseState, role: MessageRole) -> List[Message]:
        """
        Filter messages by role.
        
        Args:
            state: Current state
            role: Message role to filter by
            
        Returns:
            List of messages with specified role
        """
        return [msg for msg in state.messages if msg.role == role]
    
    @staticmethod
    def get_messages_by_node(state: BaseState, node_name: str) -> List[Message]:
        """
        Filter messages by node name.
        
        Args:
            state: Current state
            node_name: Node name to filter by
            
        Returns:
            List of messages generated by specified node
        """
        return [msg for msg in state.messages if msg.node_name == node_name]


# State Validation Utilities
class StateValidator:
    """Comprehensive validation functions for state schemas."""
    
    @staticmethod
    def validate_base_state(state: BaseState) -> bool:
        """
        Validate BaseState for consistency and requirements.
        
        Args:
            state: BaseState to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate session_id is UUID format
        try:
            UUID(state.session_id)
        except ValueError:
            raise ValueError(f"Invalid session_id format: {state.session_id}")
        
        # Validate confidence range if present
        if state.confidence is not None and not (0.0 <= state.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {state.confidence}")
        
        # Validate message count
        if len(state.messages) > MAX_CONVERSATION_HISTORY:
            raise ValueError(f"Too many messages: {len(state.messages)} > {MAX_CONVERSATION_HISTORY}")
        
        return True
    
    @staticmethod
    def validate_clarification_state(state: ClarificationState) -> bool:
        """
        Validate ClarificationState for consistency and limits.
        
        Args:
            state: ClarificationState to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # First validate base state
        StateValidator.validate_base_state(state)
        
        # Validate clarification attempts
        if state.clarification_attempts > state.max_attempts:
            raise ValueError(
                f"Clarification attempts ({state.clarification_attempts}) "
                f"exceed max_attempts ({state.max_attempts})"
            )
        
        return True
    
    @staticmethod
    def validate_state_transition(from_state: BaseState, to_state: BaseState) -> bool:
        """
        Validate state transition preserves required fields.
        
        Args:
            from_state: Source state
            to_state: Target state
            
        Returns:
            True if valid transition, raises ValueError if invalid
        """
        # Session ID must be preserved
        if from_state.session_id != to_state.session_id:
            raise ValueError("Session ID must be preserved during state transitions")
        
        # Intent should not regress (once set, should remain or improve)
        if from_state.intent and to_state.intent != from_state.intent:
            if from_state.intent != IntentType.UNKNOWN:
                raise ValueError("Intent should not change during transitions unless originally unknown")
        
        return True 


class PlanStep(BaseModel):
    """
    Individual step in an execution plan.
    """
    
    tool_name: str = Field(description="Name of the tool to execute")
    operation: str = Field(description="Operation to perform with the tool")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool operation")
    step_id: Optional[str] = Field(None, description="Unique identifier for this step")
    
    class Config:
        extra = "forbid"


class ExecutionPlan(BaseModel):
    """
    Complete execution plan with validated steps.
    """
    
    steps: List[PlanStep] = Field(default_factory=list, description="List of execution steps")
    total_steps: Optional[int] = Field(None, description="Total number of steps in the plan")
    
    @model_validator(mode='after')
    def validate_total_steps(self):
        """Set total_steps to match the number of steps."""
        self.total_steps = len(self.steps)
        return self
    
    class Config:
        extra = "forbid" 