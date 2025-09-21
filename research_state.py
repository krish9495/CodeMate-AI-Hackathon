"""
Deep Researcher Agent - LangGraph State Schema
Defines the state structure that flows through the research workflow
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass  
class ResearchSubtask:
    """Represents a subtask in the research plan"""
    question: str
    search_terms: List[str]
    priority: int = 1
    status: str = "pending"  # pending, in_progress, completed
    findings: List[DocumentChunk] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """The structured plan for researching a query"""
    original_query: str
    subtasks: List[ResearchSubtask]
    estimated_complexity: str = "medium"  # low, medium, high
    search_strategy: str = "comprehensive"


@dataclass
class Finding:
    """A synthesized finding from research"""
    content: str
    sources: List[str]
    confidence_score: float
    category: str = "general"


@dataclass
class ResearchState:
    """
    The complete state that flows through the LangGraph workflow
    This is the backbone of our research agent
    """
    # Input
    user_query: str = ""
    conversation_history: List[BaseMessage] = field(default_factory=list)
    
    # Planning
    research_plan: Optional[ResearchPlan] = None
    current_subtask_index: int = 0
    
    # Retrieval & Analysis  
    retrieved_chunks: List[DocumentChunk] = field(default_factory=list)
    all_findings: List[Finding] = field(default_factory=list)
    
    # Reasoning
    reasoning_steps: List[str] = field(default_factory=list)
    identified_gaps: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    
    # Synthesis
    synthesized_response: str = ""
    executive_summary: str = ""
    detailed_report: str = ""
    
    # System state
    current_step: str = "initialized"
    progress_percentage: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    
    # Configuration
    max_chunks_per_query: int = 20
    confidence_threshold: float = 0.7
    export_format: str = "markdown"  # markdown, pdf, json


# Helper functions for state management
def update_progress(state: ResearchState, step: str, percentage: float) -> ResearchState:
    """Update the current step and progress"""
    state.current_step = step
    state.progress_percentage = min(percentage, 100.0)
    return state


def add_finding(state: ResearchState, content: str, sources: List[str], 
                confidence: float, category: str = "general") -> ResearchState:
    """Add a new finding to the research state"""
    finding = Finding(
        content=content,
        sources=sources, 
        confidence_score=confidence,
        category=category
    )
    state.all_findings.append(finding)
    return state


def add_reasoning_step(state: ResearchState, step: str) -> ResearchState:
    """Add a reasoning step to track the thought process"""
    state.reasoning_steps.append(step)
    return state