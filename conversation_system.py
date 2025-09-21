"""
Conversation System - Manages interactive research sessions with memory
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from research_state import ResearchState
from research_workflow import DeepResearcherWorkflow
from config import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationSession:
    """Represents a research conversation session"""
    session_id: str
    started_at: datetime
    messages: List[BaseMessage] = field(default_factory=list)
    research_states: List[ResearchState] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class ConversationManager:
    """
    Manages conversation sessions and context for interactive research
    """
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.workflow = DeepResearcherWorkflow()
    
    def start_session(self, session_id: str) -> ConversationSession:
        """Start a new conversation session"""
        session = ConversationSession(
            session_id=session_id,
            started_at=datetime.now()
        )
        self.sessions[session_id] = session
        logger.info(f"Started new conversation session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing conversation session"""
        return self.sessions.get(session_id)
    
    async def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process a user message within a conversation session
        """
        session = self.get_session(session_id)
        if not session:
            session = self.start_session(session_id)
        
        # Add user message to conversation
        session.messages.append(HumanMessage(content=user_message))
        
        try:
            # Determine if this is a follow-up or new research
            is_followup = len(session.messages) > 1
            
            # Improved logic: Check if it's truly a follow-up or a new topic
            if is_followup:
                # Check if the query is about a completely new topic
                is_new_topic = self._is_new_research_topic(user_message, session.research_states)
                
                if is_new_topic:
                    # Treat as new research even if there are previous messages
                    logger.info("Detected new research topic, treating as initial query")
                    # Clear previous research context to avoid contamination
                    self._clear_research_context(session)
                    response = await self._handle_initial_query(session, user_message)
                else:
                    # Handle as follow-up question
                    response = await self._handle_followup(session, user_message)
            else:
                # Handle initial research query
                response = await self._handle_initial_query(session, user_message)
            
            # Add AI response to conversation
            session.messages.append(AIMessage(content=response["synthesized_response"]))
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = {
                "synthesized_response": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "error": True
            }
            session.messages.append(AIMessage(content=error_response["synthesized_response"]))
            return error_response
    
    async def _handle_initial_query(self, session: ConversationSession, query: str) -> Dict[str, Any]:
        """Handle the initial research query"""
        logger.info(f"Processing initial query: {query[:50]}...")
        
        # Run full research workflow
        research_state = await self.workflow.research(query, session.messages[:-1])  # Exclude current message
        session.research_states.append(research_state)
        
        # Update conversation context - handle both ResearchState object and dict
        if hasattr(research_state, 'all_findings'):
            findings_count = len(research_state.all_findings)
        else:
            findings_count = len(research_state.get('all_findings', []))
            
        session.context.update({
            "last_query": query,
            "findings_count": findings_count,
            "research_completed": True
        })
        
        # Extract response data - handle both ResearchState object and dict
        if hasattr(research_state, 'synthesized_response'):
            return {
                "synthesized_response": research_state.synthesized_response,
                "executive_summary": research_state.executive_summary,
                "follow_up_questions": research_state.follow_up_questions,
                "progress_percentage": research_state.progress_percentage,
                "findings_count": len(research_state.all_findings),
                "detailed_report": research_state.detailed_report,
                "is_followup": False
            }
        else:
            return {
                "synthesized_response": research_state.get('synthesized_response', ''),
                "executive_summary": research_state.get('executive_summary', ''),
                "follow_up_questions": research_state.get('follow_up_questions', []),
                "progress_percentage": research_state.get('progress_percentage', 100),
                "findings_count": findings_count,
                "detailed_report": research_state.get('detailed_report', ''),
                "is_followup": False
            }
    
    async def _handle_followup(self, session: ConversationSession, followup: str) -> Dict[str, Any]:
        """Handle follow-up questions with context"""
        logger.info(f"Processing follow-up: {followup[:50]}...")
        
        # Get previous research context
        last_state = session.research_states[-1] if session.research_states else None
        
        if not last_state:
            # No previous research, treat as new query
            return await self._handle_initial_query(session, followup)
        
        # Check if this is a refinement, clarification, or new direction
        followup_type = self._classify_followup(followup, last_state)
        
        if followup_type == "clarification":
            # Answer based on existing findings
            response = await self._answer_from_existing(session, followup, last_state)
        elif followup_type == "refinement":
            # Refine the original query and research more
            # Handle both ResearchState object and dict
            original_query = getattr(last_state, 'user_query', None) or last_state.get('user_query', '')
            refined_query = f"{original_query}. Specifically: {followup}"
            research_state = await self.workflow.research(refined_query, session.messages[:-1])
            session.research_states.append(research_state)
            
            # Handle both ResearchState object and dict for response
            if hasattr(research_state, 'synthesized_response'):
                response = {
                    "synthesized_response": research_state.synthesized_response,
                    "executive_summary": research_state.executive_summary,
                    "follow_up_questions": research_state.follow_up_questions,
                    "is_followup": True,
                    "followup_type": "refinement"
                }
            else:
                response = {
                    "synthesized_response": research_state.get('synthesized_response', ''),
                    "executive_summary": research_state.get('executive_summary', ''),
                    "follow_up_questions": research_state.get('follow_up_questions', []),
                    "is_followup": True,
                    "followup_type": "refinement"
                }
        else:  # new_direction
            # Treat as new research query
            return await self._handle_initial_query(session, followup)
        
        return response
    
    def _classify_followup(self, followup: str, last_state: ResearchState) -> str:
        """
        Classify the type of follow-up question
        Returns: 'clarification', 'refinement', or 'new_direction'
        """
        followup_lower = followup.lower()
        
        # Clarification indicators
        clarification_terms = [
            "what do you mean", "can you explain", "clarify", "elaborate",
            "more details", "what does this mean", "how so", "why"
        ]
        
        # Refinement indicators  
        refinement_terms = [
            "more about", "specifically", "in particular", "focus on",
            "dig deeper", "expand on", "tell me more about"
        ]
        
        # Check for clarification
        if any(term in followup_lower for term in clarification_terms):
            return "clarification"
        
        # Check for refinement
        if any(term in followup_lower for term in refinement_terms):
            return "refinement"
        
        # Check if it relates to the last query topic
        # Handle both ResearchState object and dict
        original_query = getattr(last_state, 'user_query', None) or last_state.get('user_query', '')
        last_query_words = set(original_query.lower().split())
        followup_words = set(followup_lower.split())
        
        # If significant overlap, it's likely a refinement
        overlap = len(last_query_words.intersection(followup_words))
        if overlap >= 2:
            return "refinement"
        
        # Otherwise, treat as new direction
        return "new_direction"
    
    def _is_new_research_topic(self, query: str, previous_states: List) -> bool:
        """
        Determine if the query is about a completely new research topic
        by checking for topic indicators and paper references
        """
        query_lower = query.lower()
        
        # Strong indicators of new research topic
        new_topic_indicators = [
            "paper", "document", "study about", "research on", "analysis of",
            "tell me about", "what is", "explain", "describe", "analyze"
        ]
        
        # Check for paper-specific references
        paper_references = [
            "paper 1", "paper 2", "paper 3", "paper 4", "paper 5",
            "document 1", "document 2", "document 3", "document 4", "document 5",
            "first paper", "second paper", "third paper", "fourth paper", "fifth paper"
        ]
        
        # Strong indicator: specific paper reference
        if any(ref in query_lower for ref in paper_references):
            logger.info(f"Detected paper reference in query: {query[:50]}...")
            return True
        
        # Strong indicator: new topic introduction phrases
        if any(indicator in query_lower for indicator in new_topic_indicators):
            # Check if the topic is different from previous research
            if previous_states:
                last_state = previous_states[-1]
                original_query = getattr(last_state, 'user_query', None) or last_state.get('user_query', '')
                
                # Check topic similarity
                last_query_words = set(original_query.lower().split())
                current_query_words = set(query_lower.split())
                
                # If very little overlap, likely a new topic
                overlap = len(last_query_words.intersection(current_query_words))
                overlap_ratio = overlap / max(len(current_query_words), 1)
                
                if overlap_ratio < 0.3:  # Less than 30% word overlap
                    logger.info(f"Detected new topic based on low word overlap: {overlap_ratio:.2f}")
                    return True
        
        return False

    def _clear_research_context(self, session: ConversationSession) -> None:
        """Clear previous research context when starting a new research topic"""
        logger.info("Clearing previous research context for new topic")
        
        # Clear research states to avoid contamination
        session.research_states.clear()
        
        # Clear relevant context but keep session metadata
        keys_to_clear = ['last_query', 'findings_count', 'research_completed']
        for key in keys_to_clear:
            session.context.pop(key, None)
        
        # Add marker that this is a new research session
        session.context['new_research_started'] = True
        
        logger.info("Research context cleared successfully")

    async def _answer_from_existing(self, session: ConversationSession,
                                  question: str, last_state: ResearchState) -> Dict[str, Any]:
        """Answer a clarification question using existing findings"""
        
        # Use the synthesis agent to answer based on existing findings
        from agents.synthesis_agent import SynthesisAgent
        from langchain_groq import ChatGroq
        from config import Config
        
        llm = ChatGroq(
            model=Config.GROQ_MODEL,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0.1
        )
        
        synthesis_agent = SynthesisAgent(llm)
        
        # Create a context from existing findings
        # Handle both ResearchState object and dict
        original_query = getattr(last_state, 'user_query', None) or last_state.get('user_query', '')
        context = f"Previous research on: {original_query}\n\n"
        # Handle both ResearchState object and dict for response
        synthesized_response = getattr(last_state, 'synthesized_response', None) or last_state.get('synthesized_response', '')
        context += f"Key findings: {synthesized_response[:500]}..."
        
        # Handle both ResearchState object and dict for findings
        findings = getattr(last_state, 'all_findings', []) if hasattr(last_state, 'all_findings') else last_state.get('all_findings', [])
        
        result = await synthesis_agent.synthesize_with_context(
            query=question,
            findings=findings,
            additional_context=context
        )
        
        return {
            "synthesized_response": result.get("synthesis", "I couldn't find specific information to answer that question based on the current research."),
            "is_followup": True,
            "followup_type": "clarification",
            "confidence_assessment": result.get("confidence_assessment", "medium")
        }
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get formatted conversation history"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        history = []
        for message in session.messages:
            history.append({
                "role": "human" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
                "timestamp": datetime.now().isoformat()  # In real implementation, store timestamps
            })
        
        return history
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        total_findings = sum(
            len(getattr(state, 'all_findings', []) if hasattr(state, 'all_findings') else state.get('all_findings', []))
            for state in session.research_states
        )
        
        return {
            "session_id": session_id,
            "started_at": session.started_at.isoformat(),
            "message_count": len(session.messages),
            "research_queries": len(session.research_states),
            "total_findings": total_findings,
            "is_active": session.is_active
        }