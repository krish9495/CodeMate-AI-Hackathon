"""
Synthesis Agent - Combines findings and generates comprehensive reports
"""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from research_state import Finding
from config import Config, get_logger

logger = get_logger(__name__)


class SynthesisAgent:
    """
    Advanced synthesis agent that combines findings into coherent reports
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.str_parser = StrOutputParser()
        
        # Prompt for synthesis
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research synthesizer. Your job is to combine multiple research findings into a coherent, comprehensive analysis.

Key principles:
1. Integrate findings logically and coherently
2. Acknowledge and resolve contradictions when possible
3. Highlight the most important insights
4. Maintain objectivity and cite sources
5. Draw meaningful conclusions
6. Identify practical implications

Structure your synthesis with:
- Executive Summary (2-3 sentences)
- Key Findings (organized by theme)
- Detailed Analysis 
- Conclusions and Implications
- Confidence Assessment"""),
            ("human", """Research Query: {query}

Research Findings to Synthesize:
{findings}

Reasoning Steps Taken:
{reasoning_steps}

Please create a comprehensive synthesis of these findings.""")
        ])
        
        # Prompt for report generation
        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are generating a professional research report. Create a well-structured, comprehensive report that would be suitable for academic or professional use.

Format the report with proper headings, clear organization, and professional language. Include:

1. EXECUTIVE SUMMARY
2. RESEARCH METHODOLOGY 
3. KEY FINDINGS
4. DETAILED ANALYSIS
5. CONCLUSIONS AND IMPLICATIONS
6. LIMITATIONS AND FUTURE RESEARCH
7. REFERENCES

Use markdown formatting for better readability."""),
            ("human", """Research Query: {query}

Synthesis: {synthesis}

Executive Summary: {summary}

All Findings: {all_findings}

Generate a comprehensive research report:""")
        ])
        
        # Prompt for follow-up questions
        self.followup_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the research conducted, suggest 3-5 intelligent follow-up questions that would help the user dig deeper or explore related areas.

Questions should be:
- Specific and actionable
- Build on the current findings
- Explore different perspectives or angles
- Lead to meaningful additional research

Format as a simple numbered list."""),
            ("human", """Original Query: {query}

Research Synthesis: {synthesis}

What follow-up questions would help explore this topic further?""")
        ])
    
    async def synthesize_findings(self, query: str, findings: List[Finding], 
                                reasoning_steps: List[str]) -> Dict[str, Any]:
        """
        Synthesize research findings into a coherent response
        """
        logger.info(f"Synthesizing {len(findings)} findings")
        
        if not findings:
            return {
                "response": "No findings available to synthesize.",
                "summary": "No research data found.",
                "follow_up_questions": ["Please provide more documents or refine your search query."]
            }
        
        try:
            # Format findings for synthesis
            findings_text = self._format_findings_for_synthesis(findings)
            reasoning_text = "\n".join([f"- {step}" for step in reasoning_steps])
            
            # Generate synthesis
            chain = self.synthesis_prompt | self.llm | self.str_parser
            synthesis = await chain.ainvoke({
                "query": query,
                "findings": findings_text,
                "reasoning_steps": reasoning_text
            })
            
            # Generate executive summary (extract from synthesis)
            summary = self._extract_executive_summary(synthesis)
            
            # Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(query, synthesis)
            
            result = {
                "response": synthesis,
                "summary": summary,
                "follow_up_questions": follow_up_questions
            }
            
            logger.info("Synthesis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return self._fallback_synthesis(findings)
    
    async def generate_report(self, query: str, findings: List[Finding], 
                            synthesis: str, summary: str, export_format: str = "markdown") -> str:
        """
        Generate a comprehensive research report
        """
        logger.info(f"Generating {export_format} research report")
        
        try:
            # Format all findings
            all_findings_text = self._format_all_findings(findings)
            
            # Generate comprehensive report
            chain = self.report_prompt | self.llm | self.str_parser
            report = await chain.ainvoke({
                "query": query,
                "synthesis": synthesis,
                "summary": summary,
                "all_findings": all_findings_text
            })
            
            # Add metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"""# Research Report: {query}

**Generated:** {timestamp}  
**Agent:** Deep Researcher v2.0  
**Total Findings:** {len(findings)}

---

"""
            
            full_report = header + report
            
            # Save report to file
            report_filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_path = Config.REPORTS_DIR / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            logger.info(f"Report saved to {report_path}")
            
            # Convert to other formats if requested
            if export_format == "pdf":
                return await self._convert_to_pdf(full_report, report_path.stem)
            elif export_format == "json":
                return self._convert_to_json(query, findings, synthesis, summary)
            
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return self._fallback_report(query, findings, synthesis)
    
    async def _generate_follow_up_questions(self, query: str, synthesis: str) -> List[str]:
        """Generate intelligent follow-up questions"""
        try:
            chain = self.followup_prompt | self.llm | self.str_parser
            response = await chain.ainvoke({
                "query": query,
                "synthesis": synthesis
            })
            
            # Parse numbered list
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering/bullets
                    question = line.split('.', 1)[-1].strip()
                    if question:
                        questions.append(question)
            
            return questions[:5]  # Limit to 5 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return [
                "What additional aspects of this topic should be explored?",
                "How do these findings compare to other research in the field?", 
                "What are the practical implications of these results?"
            ]
    
    def _format_findings_for_synthesis(self, findings: List[Finding]) -> str:
        """Format findings for synthesis prompt"""
        formatted = []
        
        # Group by category
        categories = {}
        for finding in findings:
            cat = finding.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(finding)
        
        # Format by category
        for category, cat_findings in categories.items():
            formatted.append(f"\n## {category.title()} Findings:")
            for i, finding in enumerate(cat_findings, 1):
                confidence = "High" if finding.confidence_score > 0.8 else "Medium" if finding.confidence_score > 0.5 else "Low"
                formatted.append(f"\n{i}. {finding.content}")
                formatted.append(f"   - Sources: {', '.join(finding.sources)}")
                formatted.append(f"   - Confidence: {confidence} ({finding.confidence_score:.2f})")
        
        return "\n".join(formatted)
    
    def _format_all_findings(self, findings: List[Finding]) -> str:
        """Format all findings for detailed report"""
        formatted = []
        for i, finding in enumerate(findings, 1):
            formatted.append(f"\n**Finding {i}:**")
            formatted.append(f"- **Content:** {finding.content}")
            formatted.append(f"- **Category:** {finding.category}")
            formatted.append(f"- **Confidence Score:** {finding.confidence_score:.2f}")
            formatted.append(f"- **Sources:** {', '.join(finding.sources)}")
        
        return "\n".join(formatted)
    
    def _extract_executive_summary(self, synthesis: str) -> str:
        """Extract executive summary from synthesis"""
        lines = synthesis.split('\n')
        
        # Look for executive summary section
        for i, line in enumerate(lines):
            if 'executive summary' in line.lower():
                # Get the next few lines
                summary_lines = []
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('#'):
                        summary_lines.append(lines[j].strip())
                    elif summary_lines and lines[j].startswith('#'):
                        break
                
                if summary_lines:
                    return ' '.join(summary_lines)
        
        # Fallback: use first paragraph
        paragraphs = [p.strip() for p in synthesis.split('\n\n') if p.strip()]
        return paragraphs[0] if paragraphs else "Executive summary not available."
    
    def _fallback_synthesis(self, findings: List[Finding]) -> Dict[str, Any]:
        """Fallback synthesis when main synthesis fails"""
        logger.info("Using fallback synthesis")
        
        # Create basic synthesis
        high_confidence = [f for f in findings if f.confidence_score > 0.7]
        
        if high_confidence:
            synthesis = f"Based on analysis of the provided documents, {len(high_confidence)} high-confidence findings were identified:\n\n"
            for i, finding in enumerate(high_confidence[:5], 1):
                synthesis += f"{i}. {finding.content}\n"
        else:
            synthesis = "The analysis identified several findings, though confidence levels vary. Further research may be needed for more definitive conclusions."
        
        return {
            "response": synthesis,
            "summary": "Analysis completed with mixed confidence levels.",
            "follow_up_questions": ["What additional sources might strengthen these findings?"]
        }
    
    def _fallback_report(self, query: str, findings: List[Finding], synthesis: str) -> str:
        """Fallback report generation"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Research Report: {query}

**Generated:** {timestamp}
**Status:** Fallback Report (Technical issues encountered)

## Summary
{synthesis}

## Key Findings
"""
        
        for i, finding in enumerate(findings[:10], 1):
            report += f"\n{i}. {finding.content} (Confidence: {finding.confidence_score:.2f})\n"
        
        return report
    
    def _convert_to_json(self, query: str, findings: List[Finding], 
                        synthesis: str, summary: str) -> str:
        """Convert report to JSON format"""
        report_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "synthesis": synthesis,
            "findings": [
                {
                    "content": f.content,
                    "category": f.category,
                    "confidence_score": f.confidence_score,
                    "sources": f.sources
                } for f in findings
            ],
            "metadata": {
                "total_findings": len(findings),
                "high_confidence_findings": len([f for f in findings if f.confidence_score > 0.8]),
                "agent_version": "Deep Researcher v2.0"
            }
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    async def _convert_to_pdf(self, content: str, filename: str) -> str:
        """Convert markdown content to PDF (placeholder for future implementation)"""
        # For now, return the markdown content
        # In a full implementation, you would use libraries like reportlab or weasyprint
        logger.info("PDF conversion not yet implemented, returning markdown")
        return content