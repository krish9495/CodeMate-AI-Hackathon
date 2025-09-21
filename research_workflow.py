"""
Deep Researcher Agent - Core LangGraph Workflow
This is the main orchestrator that coordinates all research activities
"""
import asyncio
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from research_state import ResearchState, update_progress, add_reasoning_step
from config import Config, get_logger
from groq_client import create_rate_limited_groq_client
from agents.query_analyzer import QueryAnalysisAgent
from agents.retrieval_agent import DocumentRetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.synthesis_agent import SynthesisAgent

logger = get_logger(__name__)


class DeepResearcherWorkflow:
    """
    The main LangGraph workflow that orchestrates the research process
    """
    
    def __init__(self):
        """Initialize the workflow with all agents and rate-limited LLM"""
        # Create rate-limited Groq client
        self.groq_client = create_rate_limited_groq_client()
        
        # Initialize specialized agents with rate-limited client
        self.query_analyzer = QueryAnalysisAgent(self.groq_client.llm)
        self.retrieval_agent = DocumentRetrievalAgent()
        self.reasoning_agent = ReasoningAgent(self.groq_client.llm)
        self.synthesis_agent = SynthesisAgent(self.groq_client.llm)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine for research"""
        workflow = StateGraph(ResearchState)
        
        # Define nodes (research steps)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)  
        workflow.add_node("reason_and_analyze", self._reason_and_analyze_node)
        workflow.add_node("check_completeness", self._check_completeness_node)
        workflow.add_node("synthesize_findings", self._synthesize_findings_node)
        workflow.add_node("generate_report", self._generate_report_node)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "reason_and_analyze")
        workflow.add_edge("reason_and_analyze", "check_completeness")
        
        # Conditional edge: continue research or move to synthesis
        workflow.add_conditional_edges(
            "check_completeness",
            self._should_continue_research,
            {
                "continue": "retrieve_documents",  # Loop back for more research
                "synthesize": "synthesize_findings"
            }
        )
        
        workflow.add_edge("synthesize_findings", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow
    
    # Node implementations
    async def _analyze_query_node(self, state: ResearchState) -> ResearchState:
        """Analyze user query and create research plan"""
        logger.info(f"Analyzing query: {state.user_query}")
        state = update_progress(state, "Analyzing query and creating research plan", 10)
        
        try:
            research_plan = await self.query_analyzer.analyze_query(state.user_query)
            state.research_plan = research_plan
            state = add_reasoning_step(state, f"Created research plan with {len(research_plan.subtasks)} subtasks")
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            state.error_messages.append(f"Query analysis failed: {e}")
            
        return state
    
    async def _retrieve_documents_node(self, state: ResearchState) -> ResearchState:
        """Retrieve relevant documents for current research subtask"""
        if not state.research_plan:
            return state
            
        current_subtask = state.research_plan.subtasks[state.current_subtask_index]
        logger.info(f"Retrieving documents for: {current_subtask.question}")
        
        progress = 20 + (state.current_subtask_index / len(state.research_plan.subtasks)) * 30
        state = update_progress(state, f"Retrieving documents for: {current_subtask.question}", progress)
        
        try:
            chunks = await self.retrieval_agent.retrieve_documents(
                query=current_subtask.question,
                search_terms=current_subtask.search_terms,
                max_chunks=state.max_chunks_per_query
            )
            
            current_subtask.findings.extend(chunks)
            state.retrieved_chunks.extend(chunks)
            state = add_reasoning_step(state, f"Retrieved {len(chunks)} relevant document chunks")
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            state.error_messages.append(f"Document retrieval failed: {e}")
            
        return state
    
    async def _reason_and_analyze_node(self, state: ResearchState) -> ResearchState:
        """Analyze retrieved documents and extract insights"""
        current_subtask = state.research_plan.subtasks[state.current_subtask_index]
        logger.info(f"Analyzing findings for: {current_subtask.question}")
        
        progress = 50 + (state.current_subtask_index / len(state.research_plan.subtasks)) * 20
        state = update_progress(state, f"Analyzing findings for: {current_subtask.question}", progress)
        
        try:
            findings = await self.reasoning_agent.analyze_documents(
                query=current_subtask.question,
                chunks=current_subtask.findings
            )
            
            state.all_findings.extend(findings)
            current_subtask.status = "completed"
            state = add_reasoning_step(state, f"Generated {len(findings)} insights from analysis")
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            state.error_messages.append(f"Document analysis failed: {e}")
            
        return state
    
    def _should_continue_research(self, state: ResearchState) -> str:
        """Decide whether to continue research or move to synthesis"""
        if not state.research_plan:
            return "synthesize"
        
        # Check if we've hit the maximum number of research cycles
        if len(state.reasoning_steps) > 20:  # Prevent infinite loops
            logger.info("Maximum research cycles reached, moving to synthesis")
            return "synthesize"
            
        # Move to next subtask
        state.current_subtask_index += 1
        
        # Check if we've completed all subtasks
        if state.current_subtask_index >= len(state.research_plan.subtasks):
            logger.info("All research subtasks completed, moving to synthesis")
            return "synthesize"
        
        logger.info(f"Continuing with subtask {state.current_subtask_index + 1}/{len(state.research_plan.subtasks)}")
        return "continue"
    
    async def _check_completeness_node(self, state: ResearchState) -> ResearchState:
        """Check if research is complete or if gaps need to be filled"""
        state = update_progress(state, "Checking research completeness", 70)
        
        try:
            gaps = await self.reasoning_agent.identify_gaps(
                original_query=state.user_query,
                current_findings=state.all_findings
            )
            
            state.identified_gaps = gaps
            if gaps:
                state = add_reasoning_step(state, f"Identified {len(gaps)} knowledge gaps to address")
            else:
                state = add_reasoning_step(state, "Research appears complete, no significant gaps found")
                
        except Exception as e:
            logger.error(f"Error checking completeness: {e}")
            state.error_messages.append(f"Completeness check failed: {e}")
            
        return state
    
    async def _synthesize_findings_node(self, state: ResearchState) -> ResearchState:
        """Synthesize all findings into coherent insights"""
        logger.info("Synthesizing research findings")
        state = update_progress(state, "Synthesizing findings into coherent insights", 80)
        
        try:
            synthesis_result = await self.synthesis_agent.synthesize_findings(
                query=state.user_query,
                findings=state.all_findings,
                reasoning_steps=state.reasoning_steps
            )
            
            state.synthesized_response = synthesis_result["response"]
            state.executive_summary = synthesis_result.get("summary", "")
            state.follow_up_questions = synthesis_result.get("follow_up_questions", [])
            
            state = add_reasoning_step(state, "Completed synthesis of all research findings")
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            state.error_messages.append(f"Synthesis failed: {e}")
            
        return state
    
    async def _generate_report_node(self, state: ResearchState) -> ResearchState:
        """Generate final research report"""
        logger.info("Generating final research report")
        state = update_progress(state, "Generating comprehensive research report", 95)
        
        try:
            detailed_report = await self.synthesis_agent.generate_report(
                query=state.user_query,
                findings=state.all_findings,
                synthesis=state.synthesized_response,
                summary=state.executive_summary,
                export_format=state.export_format
            )
            
            state.detailed_report = detailed_report
            state = update_progress(state, "Research completed successfully", 100)
            state = add_reasoning_step(state, "Generated comprehensive research report")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            state.error_messages.append(f"Report generation failed: {e}")
            
        return state
    
    # Public interface
    async def research(self, query: str, conversation_history: List = None) -> ResearchState:
        """Main entry point for conducting research"""
        logger.info(f"Starting research for query: {query}")
        
        # Initialize state
        initial_state = ResearchState(
            user_query=query,
            conversation_history=conversation_history or [],
            current_step="initialized"
        )
        
        # Run the workflow
        try:
            final_state = await self.app.ainvoke(initial_state)
            logger.info("Research workflow completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            initial_state.error_messages.append(f"Workflow execution failed: {e}")
            return initial_state
    
    def get_progress(self, state: ResearchState) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            "current_step": state.current_step,
            "progress_percentage": state.progress_percentage,
            "completed_subtasks": state.current_subtask_index,
            "total_subtasks": len(state.research_plan.subtasks) if state.research_plan else 0,
            "findings_count": len(state.all_findings),
            "errors": state.error_messages
        }
    
    async def initialize_embeddings(self):
        """Initialize embedding components"""
        logger.info("Initializing embeddings...")
        await self.retrieval_agent.initialize()
        
    async def process_documents_with_progress(self, documents_path: str, progress_callback):
        """Process documents with progress updates"""
        logger.info(f"Processing documents from: {documents_path}")
        
        try:
            progress_callback("Initializing document processing...")
            await self.retrieval_agent.initialize()
            
            progress_callback("Loading and chunking documents...")
            success = await self.retrieval_agent.ingest_documents(documents_path)
            
            if success:
                progress_callback("Documents processed successfully! (100%)")
                return True
            else:
                progress_callback("Failed to process documents")
                return False
                
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            progress_callback(f"Error: {e}")
            return False
    
    async def load_existing_documents_with_progress(self, progress_callback):
        """Load existing documents with progress updates"""
        logger.info("Loading existing documents...")
        
        try:
            progress_callback("Initializing document retrieval...")
            await self.retrieval_agent.initialize()
            
            progress_callback("Loading existing embeddings...")
            success = await self.retrieval_agent.load_existing()
            
            if success:
                progress_callback("Existing documents loaded successfully! (100%)")
                return True
            else:
                progress_callback("No existing documents found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
            progress_callback(f"Error: {e}")
            return False