"""
Reasoning Agent - Analyzes documents and extracts insights using Groq
"""
import json
import asyncio
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from research_state import DocumentChunk, Finding
from config import get_logger

logger = get_logger(__name__)


class ReasoningAgent:
    """
    Advanced reasoning agent that analyzes documents and generates insights
    """
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.parser = JsonOutputParser()
        
        # Prompt for document analysis
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst. Your job is to analyze the SPECIFIC document chunks provided and extract meaningful insights.

IMPORTANT: 
- ONLY analyze the content in the provided document chunks below
- DO NOT create your own examples or generic content
- Base your findings STRICTLY on the text provided in the chunks
- If the chunks don't contain relevant information, say so explicitly
- Return ONLY valid JSON with no additional text, explanations, or preamble

For each analysis, you should:
1. Identify key findings and insights FROM THE PROVIDED CHUNKS
2. Assess the confidence level of each finding
3. Note any conflicting information IN THE CHUNKS
4. Categorize findings appropriately
5. Provide source references from the chunks

Return your response as JSON:
{{
  "findings": [
    {{
      "content": "detailed finding or insight FROM THE PROVIDED CHUNKS",
      "confidence_score": 0.8,
      "category": "methodology/results/conclusion/background",
      "sources": ["source1", "source2"],
      "evidence": "specific text that supports this finding FROM THE CHUNKS"
    }}
  ],
  "conflicts": [
    {{
      "description": "description of conflicting information IN THE CHUNKS",
      "sources": ["source1", "source2"]
    }}
  ],
  "key_themes": ["theme1", "theme2"],
  "gaps_identified": ["gap1", "gap2"]
}}

Guidelines:
- Be specific and evidence-based
- Confidence score: 0.0-1.0 (1.0 = very confident)
- Only include insights that come from the provided chunks
- Reference sources accurately from the provided chunks"""),
            ("human", """Research Question: {query}

Document Chunks to Analyze:
{chunks_text}

Please analyze these documents and extract key insights related to the research question.""")
        ])
        
        # Prompt for gap identification
        self.gap_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing research completeness. Compare the original query with current findings to identify gaps.

IMPORTANT: Return ONLY valid JSON with no additional text, explanations, or preamble.

Return JSON:
{{
  "gaps": [
    {{
      "description": "what information is missing",
      "importance": "high/medium/low",
      "suggested_search": "suggested search terms or questions"
    }}
  ],
  "completeness_score": 0.75,
  "recommendation": "continue/sufficient"
}}"""),
            ("human", """Original Research Query: {query}

Current Findings:
{findings_summary}

What important information is still missing?""")
        ])
    
    async def analyze_documents(self, query: str, chunks: List[DocumentChunk]) -> List[Finding]:
        """
        Analyze document chunks and extract insights
        """
        if not chunks:
            logger.warning("No chunks provided for analysis")
            return []
        
        logger.info(f"Analyzing {len(chunks)} document chunks")
        
        try:
            # Prepare chunks text for analysis
            chunks_text = self._format_chunks_for_analysis(chunks)
            
            # Debug: Log what we're sending to the LLM
            logger.info(f"Sending query: {query[:100]}...")
            logger.info(f"Formatted chunks preview: {chunks_text[:300]}...")
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
            # Run analysis - get raw response without parser first
            chain = self.analysis_prompt | self.llm
            
            # Debug the actual parameters being sent
            params = {
                "query": query,
                "chunks_text": chunks_text
            }
            logger.info(f"Invoking with params keys: {list(params.keys())}")
            logger.info(f"Query param: {params['query'][:50]}...")
            logger.info(f"Chunks_text param length: {len(params['chunks_text'])}")
            
            response = await chain.ainvoke(params)
            
            # Parse JSON more robustly
            result = self._parse_json_response(response.content, "analysis")
            if result is None:
                logger.warning("Failed to parse JSON response, using fallback")
                return self._fallback_analysis(chunks)
            
            # Convert to Finding objects
            findings = []
            for finding_data in result.get("findings", []):
                finding = Finding(
                    content=finding_data.get("content", ""),
                    sources=finding_data.get("sources", []),
                    confidence_score=finding_data.get("confidence_score", 0.5),
                    category=finding_data.get("category", "general")
                )
                findings.append(finding)
            
            # Log conflicts and themes for debugging
            if result.get("conflicts"):
                logger.info(f"Identified {len(result['conflicts'])} conflicts in the data")
            
            if result.get("key_themes"):
                logger.info(f"Key themes identified: {', '.join(result['key_themes'])}")
            
            logger.info(f"Generated {len(findings)} insights from document analysis")
            return findings
            
        except Exception as e:
            logger.error(f"Error in document analysis: {e}")
            return self._fallback_analysis(chunks)
    
    async def identify_gaps(self, original_query: str, current_findings: List[Finding]) -> List[str]:
        """
        Identify gaps in current research findings
        """
        if not current_findings:
            return ["No findings available to analyze"]
        
        try:
            # Prepare findings summary
            findings_summary = self._format_findings_for_gap_analysis(current_findings)
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
            # Run gap analysis - get raw response without parser first
            chain = self.gap_analysis_prompt | self.llm
            response = await chain.ainvoke({
                "query": original_query,
                "findings_summary": findings_summary
            })
            
            # Parse JSON more robustly
            result = self._parse_json_response(response.content, "gap_analysis")
            if result is None:
                logger.warning("Failed to parse gap analysis JSON, returning basic gaps")
                return ["Unable to identify specific gaps due to parsing error"]
            
            # Extract gap descriptions
            gaps = []
            for gap_data in result.get("gaps", []):
                gaps.append(gap_data.get("description", "Unknown gap"))
            
            logger.info(f"Identified {len(gaps)} research gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error in gap analysis: {e}")
            return []
    
    async def synthesize_with_context(self, query: str, findings: List[Finding], 
                                    additional_context: str = "") -> Dict[str, Any]:
        """
        Synthesize findings with additional context
        """
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are synthesizing research findings into a coherent analysis.

Create a comprehensive synthesis that:
1. Integrates all findings logically
2. Resolves or acknowledges conflicts
3. Draws meaningful conclusions
4. Identifies implications

IMPORTANT: Return ONLY valid JSON with no additional text, explanations, or preamble.

Return JSON:
{{
  "synthesis": "comprehensive synthesis text",
  "key_conclusions": ["conclusion1", "conclusion2"],
  "implications": ["implication1", "implication2"],
  "confidence_assessment": "overall confidence in the synthesis",
  "limitations": ["limitation1", "limitation2"]
}}"""),
            ("human", """Research Query: {query}

Findings to Synthesize:
{findings_text}

Additional Context: {context}

Create a comprehensive synthesis:""")
        ])
        
        try:
            findings_text = "\n\n".join([
                f"Finding {i+1}: {finding.content} (Confidence: {finding.confidence_score:.2f})"
                for i, finding in enumerate(findings)
            ])
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
            chain = synthesis_prompt | self.llm
            response = await chain.ainvoke({
                "query": query,
                "findings_text": findings_text,
                "context": additional_context
            })
            
            # Parse JSON more robustly
            result = self._parse_json_response(response.content, "synthesis")
            if result is None:
                logger.warning("Failed to parse synthesis JSON, using fallback")
                return {
                    "synthesis": "Unable to parse synthesis response",
                    "key_conclusions": ["Synthesis parsing failed"],
                    "implications": ["Unable to determine implications"],
                    "confidence_assessment": "low",
                    "limitations": ["JSON parsing error in synthesis"]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return {
                "synthesis": "Unable to complete synthesis due to technical error",
                "key_conclusions": [],
                "implications": [],
                "confidence_assessment": "low",
                "limitations": ["Technical error in synthesis process"]
            }
    
    def _format_chunks_for_analysis(self, chunks: List[DocumentChunk]) -> str:
        """Format document chunks for analysis prompt"""
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"""
Chunk {i+1}:
Source: {chunk.source}
Relevance Score: {chunk.relevance_score:.3f}
Content: {chunk.content[:1000]}{'...' if len(chunk.content) > 1000 else ''}
""")
        return "\n".join(formatted)
    
    def _format_findings_for_gap_analysis(self, findings: List[Finding]) -> str:
        """Format findings for gap analysis prompt"""
        formatted = []
        for i, finding in enumerate(findings):
            formatted.append(f"""
Finding {i+1}:
Category: {finding.category}
Content: {finding.content}
Confidence: {finding.confidence_score:.2f}
Sources: {', '.join(finding.sources)}
""")
        return "\n".join(formatted)
    
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
            
            # Handle markdown code blocks (```json ... ```)
            if '```json' in response_text:
                # Match with optional whitespace/newlines after ```json
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        pass
                        
                # Also try without space after ```json (like ```json{)
                json_match = re.search(r'```json(.*?)```', response_text, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        pass
            
            # Handle code blocks without language specifier (``` ... ```)
            if '```' in response_text:
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        return json.loads(json_content)
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
            if cleaned.startswith('Based on') or 'here are' in cleaned[:50]:
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
            logger.error(f"Failed to parse JSON for {operation_type}: {response_text[:500]}...")
            logger.error(f"Full response length: {len(response_text)}")
            return None
            
        except Exception as e:
            logger.error(f"Error in JSON parsing for {operation_type}: {e}")
            return None
    
    def _fallback_analysis(self, chunks: List[DocumentChunk]) -> List[Finding]:
        """
        Fallback analysis when main analysis fails
        """
        logger.info("Using fallback analysis method")
        
        # Create basic findings from chunks
        findings = []
        for chunk in chunks[:5]:  # Limit to top 5 chunks
            finding = Finding(
                content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                sources=[chunk.source],
                confidence_score=min(chunk.relevance_score, 1.0),
                category="extracted"
            )
            findings.append(finding)
        
        return findings