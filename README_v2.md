# 🔬 Deep Researcher Agent v2.0

**Advanced AI Research Assistant with Multi-Step Reasoning**

A production-ready research agent that combines local document processing with powerful AI reasoning using LangGraph workflows and Google's Gemini AI.

## ✨ Features

### 🧠 **Intelligent Research Workflow**

- **Multi-step reasoning** with LangGraph state machines
- **Query decomposition** into research subtasks
- **Iterative knowledge gathering** with gap identification
- **Intelligent synthesis** of findings from multiple sources

### 🔍 **Advanced Document Processing**

- **Local embedding generation** using Sentence Transformers
- **FAISS vector search** for fast, accurate retrieval
- **Multiple file format support** (PDF, DOCX, TXT)
- **Optimized chunking** for better context retention

### 💬 **Interactive Conversation System**

- **Contextual follow-up questions** with conversation memory
- **Query refinement** and clarification handling
- **Real-time progress tracking** during research
- **Smart suggestions** for deeper exploration

### 📊 **Comprehensive Reporting**

- **Executive summaries** with key insights
- **Detailed research reports** with citations
- **Multiple export formats** (Markdown, JSON, PDF\*)
- **Source attribution** and confidence scores

### 🌐 **Production-Ready Interface**

- **Modern Streamlit UI** with real-time updates
- **Progress tracking** with detailed feedback
- **Error handling** and graceful fallbacks
- **Session management** with conversation history

## 🏗️ Architecture

```
User Query → Query Analysis → Research Planning → Document Retrieval →
Multi-Step Reasoning → Gap Analysis → Synthesis → Report Generation
```

### **Core Components:**

1. **LangGraph Workflow** - Orchestrates the entire research process
2. **Gemini AI Integration** - Powers reasoning, analysis, and synthesis
3. **Local Vector Store** - Fast, private document search with FAISS
4. **Conversation Manager** - Handles context and follow-up interactions
5. **Multi-Agent System** - Specialized agents for different research tasks

## 🚀 Quick Start

### **1. Setup**

```bash
git clone <your-repo>
cd "Codemate AI"

# Install dependencies (in your virtual environment)
pip install -r requirements.txt
```

### **2. Configuration**

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### **3. Launch**

```bash
python launch.py
```

Or directly:

```bash
streamlit run deep_researcher_app.py
```

## 📖 Usage Guide

### **Document Setup**

1. Upload PDF, DOCX, or TXT files
2. System processes and creates embeddings locally
3. Documents are chunked and indexed for optimal retrieval

### **Research Process**

1. **Ask a question**: "What are the latest trends in AI sustainability?"
2. **System analyzes** and creates research plan
3. **Retrieves relevant** document sections
4. **Reasons through** findings step-by-step
5. **Synthesizes** comprehensive response

### **Interactive Features**

- **Follow-up questions**: "Tell me more about energy efficiency"
- **Clarifications**: "What do you mean by 'sustainable AI'?"
- **Refinements**: "Focus specifically on data center optimization"

## 🔧 Configuration Options

### **Environment Variables**

- `GEMINI_API_KEY` - Your Google AI API key (required)
- `GEMINI_MODEL` - Gemini model to use (default: gemini-1.5-pro)
- `EMBEDDING_MODEL` - Sentence transformer model (default: all-MiniLM-L6-v2)
- `CHUNK_SIZE` - Document chunk size (default: 1200)
- `CHUNK_OVERLAP` - Chunk overlap (default: 150)

### **Performance Tuning**

- `BATCH_SIZE` - Embedding batch size (default: 8)
- `MAX_CHUNKS_PER_QUERY` - Max chunks retrieved per query (default: 20)
- `CONFIDENCE_THRESHOLD` - Minimum confidence for findings (default: 0.7)

## 🏛️ System Architecture

### **LangGraph Workflow**

```python
StateGraph({
    "analyze_query": QueryAnalysisAgent,
    "retrieve_documents": DocumentRetrievalAgent,
    "reason_and_analyze": ReasoningAgent,
    "check_completeness": CompletenessChecker,
    "synthesize_findings": SynthesisAgent,
    "generate_report": ReportGenerator
})
```

### **Agent Specialization**

- **Query Analyzer**: Breaks queries into research subtasks
- **Retrieval Agent**: Enhanced multi-strategy document search
- **Reasoning Agent**: Analyzes documents and extracts insights
- **Synthesis Agent**: Combines findings into coherent narratives

### **State Management**

- **ResearchState**: Tracks query, plan, findings, and progress
- **Conversation Memory**: Maintains context across interactions
- **Progress Tracking**: Real-time feedback on research status

## 📊 Performance Features

### **Optimized Processing**

- **Async operations** for concurrent processing
- **Batch embedding** generation for efficiency
- **Smart caching** of frequently accessed data
- **Progressive loading** with real-time feedback

### **Scalability**

- **Stateless agents** for easy horizontal scaling
- **Local embeddings** reduce API costs and latency
- **Configurable batch sizes** for different hardware
- **Error recovery** with graceful degradation

## 🔒 Privacy & Security

- **Local document processing** - Files never leave your system
- **Private embeddings** - Generated locally with Sentence Transformers
- **Secure API calls** - Only processed insights sent to Gemini
- **No data persistence** in external services

## 🚀 Deployment Options

### **Local Development**

```bash
python launch.py
```

### **Docker Deployment**

```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "deep_researcher_app.py"]
```

### **Cloud Deployment**

- **Streamlit Cloud**: Connect your GitHub repo
- **Railway/Render**: Deploy with Docker
- **AWS/GCP**: Use container services

## 🛠️ Development

### **Project Structure**

```
├── deep_researcher_app.py      # Main Streamlit interface
├── research_workflow.py        # Core LangGraph workflow
├── research_state.py          # State management schemas
├── conversation_system.py     # Interactive conversation handling
├── config.py                 # Configuration management
├── agents/                   # Specialized research agents
│   ├── query_analyzer.py    # Query analysis and planning
│   ├── retrieval_agent.py   # Document retrieval
│   ├── reasoning_agent.py   # Document analysis
│   └── synthesis_agent.py   # Findings synthesis
├── app.py                   # Legacy orchestrator (for compatibility)
└── launch.py               # Application launcher
```

### **Adding New Features**

1. **New Agents**: Create in `agents/` directory
2. **Workflow Nodes**: Add to `research_workflow.py`
3. **State Fields**: Extend `ResearchState` in `research_state.py`
4. **UI Components**: Modify `deep_researcher_app.py`

## 📈 Monitoring & Analytics

### **Built-in Metrics**

- Research query completion rates
- Average processing times
- User interaction patterns
- System performance metrics

### **Logging**

- Structured logging with configurable levels
- Research workflow tracking
- Error monitoring and alerting
- Performance profiling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 🆘 Troubleshooting

### **Common Issues**

**"GEMINI_API_KEY not found"**

- Ensure `.env` file exists with valid API key
- Check environment variable loading

**"No documents available for retrieval"**

- Upload and process documents first
- Check if FAISS index was created successfully

**"Import error: langgraph"**

- Install missing dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated

**Slow processing**

- Reduce `CHUNK_SIZE` or `MAX_CHUNKS_PER_QUERY`
- Check CPU/memory usage during embedding generation

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain/LangGraph** for workflow orchestration
- **Google AI** for Gemini API access
- **Sentence Transformers** for local embeddings
- **FAISS** for efficient vector search
- **Streamlit** for the web interface

---

**Deep Researcher Agent v2.0** - Transforming how you research with AI 🚀
