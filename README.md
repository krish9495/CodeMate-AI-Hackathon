# Deep Researcher Agent v2.0

## ⚠️ Security Notice

**Before running the application, you MUST set up your environment variables:**

1. Create a `.env` file in the project root
2. Add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Setup Instructions

1. **Clone the repository**
2. **Create virtual environment**:
   ```bash
   python -m venv myenv
   myenv/Scripts/activate  # On Windows
   # or
   source myenv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables** (create `.env` file with your API keys)
5. **Run the application**:
   ```bash
   streamlit run deep_researcher_app.py
   ```

## Features

- 🔬 Advanced AI research with document analysis
- 📄 PDF export functionality
- 💬 Interactive conversation system
- 📊 Multi-document research synthesis
- 🎯 Context-aware follow-up questions

## Required API Keys

- **Groq API Key**: Required for AI model access
- Get your free API key at: https://console.groq.com/

## Note

The `.env` file is not included in this repository for security reasons. You must create it yourself with your own API keys.
