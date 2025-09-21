"""
Deep Researcher Agent - Production Streamlit Interface
Advanced research assistant with LangGraph workflow and Gemini AI
"""
import streamlit as st
import asyncio
import uuid
from pathlib import Path
import tempfile
import shutil
import json
import re
from datetime import datetime
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="Deep Researcher Agent v2.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from config import Config, setup_logging
    from conversation_system import ConversationManager
    from research_workflow import DeepResearcherWorkflow
    
    # Set up logging
    setup_logging()
    
    # Validate configuration
    if not Config.validate_config():
        st.error("Configuration validation failed. Please check your .env file.")
        st.stop()
        
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


class StreamlitInterface:
    """Production-ready Streamlit interface for the Deep Researcher Agent"""
    
    def __init__(self):
        # Initialize session state
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if "conversation_manager" not in st.session_state:
            st.session_state.conversation_manager = ConversationManager()
        
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = False
        
        if "current_research" not in st.session_state:
            st.session_state.current_research = None
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """Main application interface"""
        
        # Header
        st.title("🔬 Deep Researcher Agent v2.0")
        st.markdown("*Advanced AI Research Assistant with Multi-Step Reasoning*")
        
        # Sidebar for configuration and status
        with st.sidebar:
            self.render_sidebar()
        
        # Main content area
        if not st.session_state.documents_processed:
            self.render_document_upload()
        else:
            self.render_research_interface()
    
    def render_sidebar(self):
        """Render the sidebar with system status and configuration"""
        st.header("📊 System Status")
        
        # Configuration status
        with st.expander("🔧 Configuration"):
            if Config.GROQ_API_KEY:
                st.success("✅ Groq API configured")
            else:
                st.error("❌ Groq API key not found")
            
            st.info(f"📝 Model: {Config.GROQ_MODEL}")
            st.info(f"🔤 Embeddings: {Config.EMBEDDING_MODEL}")
        
        # Document status
        with st.expander("📄 Documents"):
            if st.session_state.documents_processed:
                st.success("✅ Documents indexed and ready")
                
                # Show document stats
                try:
                    from agents.retrieval_agent import DocumentRetrievalAgent
                    retrieval_agent = DocumentRetrievalAgent()
                    stats = retrieval_agent.get_retrieval_stats()
                    
                    if stats.get("status") == "ready":
                        st.info(f"📊 {stats.get('total_chunks', 0)} chunks indexed")
                    
                except Exception as e:
                    st.warning(f"Could not load document stats: {e}")
            else:
                st.warning("⏳ Documents not yet processed")
                
        # Session info
        with st.expander("💬 Session Info"):
            session = st.session_state.conversation_manager.get_session(st.session_state.session_id)
            if session:
                st.info(f"🆔 Session: {st.session_state.session_id[:8]}...")
                st.info(f"⏰ Started: {session.started_at.strftime('%H:%M')}")
                st.info(f"💬 Messages: {len(session.messages)}")
        
        # Reset options
        st.header("🔄 Reset Options")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        
        if st.button("📄 Reset Documents"):
            st.session_state.documents_processed = False
            st.rerun()
    
    def render_document_upload(self):
        """Render the document upload and processing interface"""
        st.header("📄 Document Setup")
        st.markdown("Upload your research documents to get started.")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or TXT files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload the documents you want to research. The system will create embeddings for intelligent search."
        )
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            use_existing = st.checkbox(
                "Use existing processed documents",
                help="Use previously processed documents if available"
            )
        
        with col2:
            chunk_size = st.selectbox(
                "Chunk Size",
                options=[800, 1200, 1600],
                index=1,
                help="Size of text chunks for processing"
            )
        
        # Process documents button
        if st.button("🚀 Process Documents", type="primary"):
            if uploaded_files or use_existing:
                import asyncio
                asyncio.run(self.process_documents(uploaded_files, use_existing, chunk_size))
            else:
                st.error("Please upload files or select to use existing documents.")
    
    async def process_documents(self, uploaded_files: List, use_existing: bool, chunk_size: int):
        """Process documents with real-time progress feedback"""
        
        # Create progress components
        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        def update_progress(message: str):
            progress_container.text(message)
            # Extract percentage if available
            if "%" in message:
                try:
                    pct_str = message.split("(")[1].split("%")[0]
                    pct = float(pct_str) / 100
                    progress_bar.progress(min(pct, 1.0))
                except:
                    pass
        
        try:
            if not use_existing and uploaded_files:
                # Save uploaded files
                temp_dir = Path(tempfile.mkdtemp())
                update_progress(f"Saving {len(uploaded_files)} files...")
                
                for uploaded_file in uploaded_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.read())
                
                # Process documents using new workflow system
                from research_workflow import DeepResearcherWorkflow
                workflow = DeepResearcherWorkflow()
                
                # Initialize the workflow
                update_progress("Initializing document processor...")
                await workflow.initialize_embeddings()
                
                # Process documents with progress callback
                success = await workflow.process_documents_with_progress(str(temp_dir), update_progress)
                
                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)
                
            else:
                # Use existing documents
                update_progress("Loading existing documents...")
                from research_workflow import DeepResearcherWorkflow
                workflow = DeepResearcherWorkflow()
                await workflow.initialize_embeddings()
                success = await workflow.load_existing_documents_with_progress(update_progress)
            
            if success:
                progress_container.text("✅ Documents processed successfully!")
                progress_bar.progress(1.0)
                st.session_state.documents_processed = True
                status_container.success("🎉 Ready to research! The interface will refresh in 2 seconds...")
                
                # Auto-refresh after success
                import time
                time.sleep(2)
                st.rerun()
            else:
                status_container.error("❌ Document processing failed. Please check the logs.")
                
        except Exception as e:
            status_container.error(f"❌ Error processing documents: {e}")
            progress_container.text(f"Error: {e}")
    
    def render_research_interface(self):
        """Render the main research interface"""
        
        # Research header
        st.header("🔍 Research Assistant")
        
        # Create two columns for chat and results
        chat_col, results_col = st.columns([1, 1])
        
        with chat_col:
            st.subheader("💬 Chat Interface")
            self.render_chat_interface()
        
        with results_col:
            st.subheader("📊 Research Results")
            self.render_results_panel()
    
    def render_chat_interface(self):
        """Render the chat interface"""
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
                        # Show additional info for assistant messages
                        if message["role"] == "assistant" and message.get("metadata"):
                            with st.expander("ℹ️ Research Details"):
                                metadata = message["metadata"]
                                if metadata.get("findings_count"):
                                    st.info(f"📊 Findings: {metadata['findings_count']}")
                                if metadata.get("follow_up_questions"):
                                    st.info("💡 **Suggested follow-ups:**")
                                    for q in metadata["follow_up_questions"][:3]:
                                        if st.button(f"🔍 {q}", key=f"followup_{hash(q)}"):
                                            self.handle_user_input(q)
        
        # Chat input
        user_input = st.chat_input("Ask a research question...")
        
        if user_input:
            self.handle_user_input(user_input)
        
        # Quick action buttons
        st.markdown("**💡 Quick Actions:**")
        quick_actions = [
            "Summarize the key findings",
            "What are the main conclusions?", 
            "Generate a detailed report"
        ]
        
        cols = st.columns(len(quick_actions))
        for i, action in enumerate(quick_actions):
            with cols[i]:
                if st.button(action, key=f"quick_{i}"):
                    self.handle_user_input(action)
    
    def handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show thinking indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Researching your question..."):
                
                # Process the message asynchronously
                try:
                    # Run the async research process
                    response = asyncio.run(
                        st.session_state.conversation_manager.process_message(
                            st.session_state.session_id,
                            user_input
                        )
                    )
                    
                    # Display the response
                    st.markdown(response["synthesized_response"])
                    
                    # Add to chat history with metadata
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response["synthesized_response"],
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "findings_count": response.get("findings_count", 0),
                            "follow_up_questions": response.get("follow_up_questions", []),
                            "is_followup": response.get("is_followup", False)
                        }
                    })
                    
                    # Update current research for results panel
                    st.session_state.current_research = response
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_msg)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Rerun to update the interface
        st.rerun()
    
    def render_results_panel(self):
        """Render the results panel with detailed information"""
        
        if not st.session_state.current_research:
            st.info("💬 Start a research conversation to see detailed results here.")
            return
        
        research = st.session_state.current_research
        
        # Executive Summary
        if research.get("executive_summary"):
            with st.expander("📋 Executive Summary", expanded=True):
                st.markdown(research["executive_summary"])
        
        # Research Metrics
        with st.expander("📊 Research Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Findings", research.get("findings_count", 0))
                
            with col2:
                progress = research.get("progress_percentage", 0)
                st.metric("Completeness", f"{progress:.0f}%")
        
        # Follow-up Questions
        if research.get("follow_up_questions"):
            with st.expander("💡 Suggested Follow-up Questions"):
                for i, question in enumerate(research["follow_up_questions"]):
                    if st.button(f"🔍 {question}", key=f"result_followup_{i}"):
                        self.handle_user_input(question)
        
        # Export Options
        with st.expander("📤 Export Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Download Markdown"):
                    if research.get("detailed_report"):
                        st.download_button(
                            "Download Report",
                            research["detailed_report"],
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
            
            with col2:
                if st.button("📊 Download JSON"):
                    # Create JSON export
                    export_data = {
                        "query": "Last research query",  # Would need to track this
                        "timestamp": datetime.now().isoformat(),
                        "response": research.get("synthesized_response", ""),
                        "summary": research.get("executive_summary", ""),
                        "findings_count": research.get("findings_count", 0)
                    }
                    
                    st.download_button(
                        "Download JSON",
                        json.dumps(export_data, indent=2),
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("📄 Download PDF"):
                    if research.get("detailed_report"):
                        pdf_data = self.generate_pdf_report(research)
                        st.download_button(
                            "Download PDF",
                            pdf_data,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
    
    def generate_pdf_report(self, research_data: Dict[str, Any]) -> bytes:
        """Generate a PDF report from research data"""
        # Import the standalone PDF generator
        from pdf_generator import generate_pdf_report
        return generate_pdf_report(research_data)
        import re
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#A23B72')
        )
        
        # Build content
        story = []
        
        # Title
        title = "🔬 Deep Research Report"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # Metadata table
        metadata = [
            ['Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Findings Count:', str(research_data.get('findings_count', 0))],
            ['Completeness:', f"{research_data.get('progress_percentage', 100):.0f}%"]
        ]
        
        metadata_table = Table(metadata, colWidths=[1.5*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8F9FA')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E9ECEF'))
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 30))
        
        # Executive Summary
        if research_data.get('executive_summary'):
            story.append(Paragraph("📋 Executive Summary", heading_style))
            # Clean and format the summary text
            summary_text = self._clean_text_for_pdf(research_data['executive_summary'])
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Main Research Content
        if research_data.get('synthesized_response'):
            story.append(Paragraph("🔍 Research Analysis", heading_style))
            # Clean and format the main content
            main_content = self._clean_text_for_pdf(research_data['synthesized_response'])
            story.append(Paragraph(main_content, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Follow-up Questions
        if research_data.get('follow_up_questions'):
            story.append(Paragraph("💡 Suggested Follow-up Questions", heading_style))
            for i, question in enumerate(research_data['follow_up_questions'][:5], 1):
                clean_question = self._clean_text_for_pdf(question)
                story.append(Paragraph(f"{i}. {clean_question}", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Detailed Report (if different from synthesized_response)
        if research_data.get('detailed_report') and research_data['detailed_report'] != research_data.get('synthesized_response'):
            story.append(Paragraph("📊 Detailed Findings", heading_style))
            detailed_text = self._clean_text_for_pdf(research_data['detailed_report'])
            story.append(Paragraph(detailed_text, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 50))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Generated by Deep Researcher Agent v2.0", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        buffer.seek(0)
        return buffer.read()
    
    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean text for PDF generation by removing problematic characters and formatting"""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
        
        # Remove markdown headers (convert to bold)
        text = re.sub(r'^#{1,6}\s*(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
        
        # Clean up bullet points
        text = re.sub(r'^[-*+]\s*', '• ', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = text.strip()
        
        # Replace problematic characters for ReportLab
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Restore the formatting tags we want to keep
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        text = text.replace('&lt;font name="Courier"&gt;', '<font name="Courier">').replace('&lt;/font&gt;', '</font>')
        
        return text


def main():
    """Main application entry point"""
    app = StreamlitInterface()
    app.run()


if __name__ == "__main__":
    main()