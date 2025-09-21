# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Visit**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Repository settings**:
   - Repository: `krish9495/CodeMate-AI-Hackathon`
   - Branch: `main`
   - Main file path: `deep_researcher_app.py`

5. **Environment Variables** (Click "Advanced settings" → "Secrets"):
```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "gemma2-9b-it"
```

6. **Click "Deploy!"**

## Features Available in Cloud Deployment:
- ✅ Document upload and processing
- ✅ AI-powered research with Groq API
- ✅ Interactive conversations and follow-up questions
- ✅ PDF, Markdown, and JSON export
- ✅ Multi-document context switching
- ✅ Real-time progress tracking

## Requirements:
- Groq API key (get one free at https://groq.com/)
- All dependencies are automatically installed from `requirements.txt`

Your app will be available at: `https://your-app-name.streamlit.app/`