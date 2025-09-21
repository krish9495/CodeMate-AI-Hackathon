"""
Configuration and utilities for the Deep Researcher Agent
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

class Config:
    """Central configuration for the research agent"""
    
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Must be set in .env file
    GROQ_MODEL = os.getenv("GROQ_MODEL", "gemma2-9b-it")  # Higher token limit: 15K/min vs 6K/min
    
    # Token Conservation - ULTRA conservative to stay under 10K tokens/minute
    GROQ_MAX_TOKENS = 150  # Reduced further for rate limiting
    GROQ_TEMPERATURE = 0.1
    
    # Embedding Configuration  
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    RESEARCH_DATA_DIR = BASE_DIR / "research_data"
    DOCUMENTS_PATH = RESEARCH_DATA_DIR / "documents.json"
    EMBEDDINGS_DIR = RESEARCH_DATA_DIR / "embeddings"
    REPORTS_DIR = RESEARCH_DATA_DIR / "reports"
    
    # Research Configuration
    MAX_SUBTASKS = 5
    MAX_CHUNKS_PER_QUERY = 20
    CONFIDENCE_THRESHOLD = 0.7
    MAX_RETRIEVAL_ITERATIONS = 3
    
    # Performance
    BATCH_SIZE = 8
    MAX_CONCURRENT_REQUESTS = 3
    
    # Rate Limiting and Retry Configuration - ULTRA CONSERVATIVE
    GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "5"))
    GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "120"))
    GROQ_RETRY_DELAY = float(os.getenv("GROQ_RETRY_DELAY", "60.0"))  # 60 second base delay
    GROQ_MAX_BACKOFF = float(os.getenv("GROQ_MAX_BACKOFF", "300.0"))  # 5 minute max backoff
    GROQ_REQUESTS_PER_MINUTE = int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "1"))  # Only 1 request per minute
    MIN_REQUEST_INTERVAL = 90  # Minimum 90 seconds between any requests
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.GROQ_API_KEY:
            logging.error("GROQ_API_KEY not found in environment variables")
            return False
        
        # Ensure directories exist
        cls.RESEARCH_DATA_DIR.mkdir(exist_ok=True)
        cls.EMBEDDINGS_DIR.mkdir(exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)
        
        return True


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Config.BASE_DIR / "research_agent.log")
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    return logging.getLogger(name)