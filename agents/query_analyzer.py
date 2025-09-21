"""
Query Analysis Agent - Analyzes user queries and             ("human", "Analyze this research query and create a comprehensive research plan:\\n\\n{query}")reates research plans
"""
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from research_state import ResearchPlan, ResearchSubtask
from config import get_logger

logger = get_logger(__name__)


class QueryAnalysisAgent:
    """
    Specialized agent that analyzes user queries and creates structured research plans
    """
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.parser = JsonOutputParser()
        
        # Prompt template for query analysis
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research strategist. Your job is to analyze user queries and create comprehensive research plans.

For each query, you should:
1. Break down the query into specific research subtasks
2. Generate relevant search terms for each subtask
3. Prioritize subtasks by importance
4. Estimate the complexity of the research

Return your response as JSON with this structure:
{{
  "original_query": "the user's original question",
  "estimated_complexity": "low/medium/high",
  "search_strategy": "comprehensive/focused/exploratory",
  "subtasks": [
    {{
      "question": "specific research question",
      "search_terms": ["term1", "term2", "term3"],
      "priority": 1
    }}
  ]
}}

Guidelines:
- Create 2-5 subtasks maximum
- Each subtask should be specific and focused
- Include synonyms and related terms in search_terms
- Higher priority number = more important
- Consider different aspects, perspectives, and angles"""),
            ("human", "Analyze this research query and create a comprehensive research plan:\n\n{query}")
        ])
    
    async def analyze_query(self, query: str) -> ResearchPlan:
        """
        Analyze a user query and create a structured research plan
        """
        logger.info(f"Analyzing query: {query[:100]}...")
        
        try:
            # Generate research plan using Groq
            chain = self.analysis_prompt | self.llm
            response = await chain.ainvoke({"query": query})
            
            # Parse JSON more robustly
            result = self._parse_json_response(response.content, "query_analysis")
            if result is None:
                logger.warning("Failed to parse JSON response, using fallback")
                return self._create_fallback_plan(query)
            
            # Create subtasks
            subtasks = []
            for subtask_data in result.get("subtasks", []):
                subtask = ResearchSubtask(
                    question=subtask_data.get("question", ""),
                    search_terms=subtask_data.get("search_terms", []),
                    priority=subtask_data.get("priority", 1)
                )
                subtasks.append(subtask)
            
            # Create research plan
            research_plan = ResearchPlan(
                original_query=query,
                subtasks=subtasks,
                estimated_complexity=result.get("estimated_complexity", "medium"),
                search_strategy=result.get("search_strategy", "comprehensive")
            )
            
            logger.info(f"Created research plan with {len(subtasks)} subtasks")
            return research_plan
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Fallback: create a simple research plan
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> ResearchPlan:
        """
        Create a simple fallback research plan if the main analysis fails
        """
        logger.info("Creating fallback research plan")
        
        # Extract potential keywords from the query
        words = query.lower().split()
        search_terms = [word for word in words if len(word) > 3]
        
        # Create a single subtask
        subtask = ResearchSubtask(
            question=query,
            search_terms=search_terms[:10],  # Limit to 10 terms
            priority=1
        )
        
        return ResearchPlan(
            original_query=query,
            subtasks=[subtask],
            estimated_complexity="medium",
            search_strategy="comprehensive"
        )
    
    async def refine_query(self, original_query: str, context: str, user_feedback: str) -> str:
        """
        Refine a query based on context and user feedback
        """
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are helping refine research queries based on context and user feedback.
            
Given:
- Original query
- Research context found so far
- User feedback

Create a refined, more specific query that will yield better results.
Return only the refined query, nothing else."""),
            ("human", """Original query: {original_query}

Context: {context}

User feedback: {user_feedback}

Refined query:""")
        ])
        
        try:
            chain = refinement_prompt | self.llm
            response = await chain.ainvoke({
                "original_query": original_query,
                "context": context,
                "user_feedback": user_feedback
            })
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return original_query
    
    def _parse_json_response(self, response_text: str, operation_type: str) -> dict:
        """
        Robustly parse JSON response from LLM, handling formatting issues
        """
        try:
            import json
            import re
            
            # First, try direct parsing
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # Try to find JSON block in the response
            # Look for content between first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx+1]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            
            # Try to clean up the response text
            cleaned = response_text.strip()
            
            # Remove any leading text before JSON
            if 'Here\'s' in cleaned[:50] or 'Since you' in cleaned[:50]:
                # Find where JSON likely starts
                json_start = cleaned.find('{')
                if json_start > 0:
                    cleaned = cleaned[json_start:]
            
            # Try parsing cleaned version
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
                
            # Log the problematic response for debugging
            logger.error(f"Failed to parse JSON for {operation_type}: {response_text[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error in JSON parsing for {operation_type}: {e}")
            return None