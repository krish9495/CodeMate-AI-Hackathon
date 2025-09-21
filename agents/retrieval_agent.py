"""
Document Retrieval Agent - Enhanced retrieval with multiple strategies
"""
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from research_state import DocumentChunk
from config import Config, get_logger

logger = get_logger(__name__)


class DocumentRetrievalAgent:
    """
    Enhanced document retrieval agent with multiple search strategies
    """
    
    def __init__(self):
        # Load embedding model and FAISS index
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunk_metadata = []
        self.documents = {}
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self) -> bool:
        """Load existing FAISS index and metadata"""
        try:
            # Try new path first (research_data directory)
            index_path = Config.RESEARCH_DATA_DIR / "document_index.faiss"
            metadata_path = Config.RESEARCH_DATA_DIR / "chunk_metadata.json"
            
            if index_path.exists() and metadata_path.exists():
                import json
                
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'r') as f:
                    metadata_list = json.load(f)
                    # Convert the format to match DocumentChunk
                    self.chunk_metadata = []
                    for chunk in metadata_list:
                        doc_chunk = DocumentChunk(
                            content=chunk["text"],  # Map 'text' to 'content'
                            source=chunk.get("source", ""),
                            chunk_id=chunk.get("chunk_id", ""),
                            metadata={}
                        )
                        self.chunk_metadata.append(doc_chunk)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} chunks")
                return True
            
            # Fallback to old path
            old_index_path = Config.EMBEDDINGS_DIR / "faiss_index.bin"
            old_metadata_path = Config.EMBEDDINGS_DIR / "faiss_metadata.pkl"
            
            if old_index_path.exists() and old_metadata_path.exists():
                import pickle
                
                self.index = faiss.read_index(str(old_index_path))
                with open(old_metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} chunks")
                return True
            else:
                logger.warning("No existing FAISS index found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    async def retrieve_documents(self, query: str, search_terms: List[str], 
                               max_chunks: int = 20) -> List[DocumentChunk]:
        """
        Retrieve relevant documents using multiple strategies
        """
        # Debug logging
        logger.info(f"Retrieve called with query: {query[:50]}...")
        logger.info(f"Index status: {self.index is not None}, ntotal: {self.index.ntotal if self.index else 'N/A'}")
        logger.info(f"Metadata count: {len(self.chunk_metadata)}")
        
        if not self.index or self.index.ntotal == 0:
            logger.warning("No documents available for retrieval")
            logger.info(f"Index state: {self.index is not None}, ntotal: {self.index.ntotal if self.index else 'None'}")
            logger.info(f"Metadata state: {len(self.chunk_metadata)} items")
            
            # Try reloading the index
            logger.info("Attempting to reload index...")
            if self._load_index():
                logger.info(f"Index reloaded successfully with {self.index.ntotal} chunks")
            else:
                logger.error("Failed to reload index")
                return []
        
        logger.info(f"Starting retrieval with {self.index.ntotal} chunks available")
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        try:
            # Strategy 1: Direct query search
            direct_chunks = await self._semantic_search(query, max_chunks // 2)
            
            # Strategy 2: Search with expanded terms
            expanded_chunks = []
            if search_terms:
                for term in search_terms[:3]:  # Limit to top 3 terms
                    term_chunks = await self._semantic_search(term, max_chunks // 4)
                    expanded_chunks.extend(term_chunks)
            
            # Combine and deduplicate results
            all_chunks = direct_chunks + expanded_chunks
            unique_chunks = self._deduplicate_chunks(all_chunks)
            
            # Re-rank by relevance to original query
            final_chunks = await self._rerank_chunks(query, unique_chunks, max_chunks)
            
            logger.info(f"Retrieved {len(final_chunks)} relevant document chunks")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []
    
    async def _semantic_search(self, query: str, k: int) -> List[DocumentChunk]:
        """Perform semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), min(k, self.index.ntotal))
            
            # Convert results to DocumentChunk objects
            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.chunk_metadata):
                    continue
                    
                metadata = self.chunk_metadata[idx]
                chunk = DocumentChunk(
                    content=metadata.content,
                    source=metadata.source,
                    chunk_id=metadata.chunk_id,
                    metadata=metadata.metadata,
                    relevance_score=float(score)
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove duplicate chunks based on chunk_id"""
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    async def _rerank_chunks(self, query: str, chunks: List[DocumentChunk], 
                           max_results: int) -> List[DocumentChunk]:
        """Re-rank chunks by relevance to the original query"""
        if not chunks:
            return []
        
        try:
            # Generate embeddings for query and all chunk contents
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts, normalize_embeddings=True)
            
            # Calculate cosine similarities
            similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
            
            # Update relevance scores and sort
            for i, chunk in enumerate(chunks):
                chunk.relevance_score = float(similarities[i])
            
            # Sort by relevance and return top results
            chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            return chunks[:max_results]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback: return original chunks sorted by existing score
            chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            return chunks[:max_results]
    
    async def expand_query(self, query: str) -> List[str]:
        """Generate expanded search terms for the query"""
        # Simple keyword extraction - can be enhanced with more sophisticated methods
        words = query.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add some variations (this is simplified - could use NLP libraries for better expansion)
        expanded = keywords.copy()
        
        return expanded[:10]  # Limit to 10 terms
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        if not self.index:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "total_chunks": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": type(self.index).__name__
        }
    
    async def initialize(self):
        """Initialize the retrieval agent"""
        logger.info("Initializing document retrieval agent...")
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        return True
    
    async def ingest_documents(self, documents_path: str) -> bool:
        """Ingest documents from the specified path"""
        from pathlib import Path
        import os
        
        try:
            doc_path = Path(documents_path)
            if not doc_path.exists():
                logger.error(f"Documents path does not exist: {documents_path}")
                return False
            
            # Find documents
            doc_files = []
            if doc_path.is_file():
                doc_files = [doc_path]
            else:
                for ext in ['*.pdf', '*.docx', '*.txt']:
                    doc_files.extend(doc_path.glob(ext))
                    doc_files.extend(doc_path.glob(f'**/{ext}'))
            
            if not doc_files:
                logger.warning(f"No documents found in {documents_path}")
                return False
            
            logger.info(f"Processing {len(doc_files)} documents...")
            
            # Process each document
            all_chunks = []
            for doc_file in doc_files:
                chunks = await self._process_single_document(str(doc_file))
                all_chunks.extend(chunks)
            
            if all_chunks:
                # Build FAISS index
                await self._build_index(all_chunks)
                logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(doc_files)} documents")
                return True
            else:
                logger.warning("No text chunks extracted from documents")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            return False
    
    async def _process_single_document(self, file_path: str) -> List[Dict]:
        """Process a single document file"""
        import pdfplumber
        from docx import Document
        
        try:
            file_path = Path(file_path)
            text = ""
            
            if file_path.suffix.lower() == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            
            elif file_path.suffix.lower() == '.docx':
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                    
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Split into chunks
            chunks = await self._split_text_into_chunks(text, str(file_path))
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return []
    
    async def _split_text_into_chunks(self, text: str, source: str) -> List[Dict]:
        """Split text into chunks for embedding"""
        if not text.strip():
            return []
        
        chunk_size = Config.CHUNK_SIZE
        chunk_overlap = Config.CHUNK_OVERLAP
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Only keep meaningful chunks
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': f"{Path(source).stem}_chunk_{len(chunks)}",
                    'word_count': len(chunk_words)
                })
        
        return chunks
    
    async def _build_index(self, chunks: List[Dict]):
        """Build FAISS index from document chunks"""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index (cosine similarity)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunk_metadata = chunks
        
        # Save index and metadata
        await self._save_index()
        
    async def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            from pathlib import Path
            import json
            
            data_dir = Path("research_data")
            data_dir.mkdir(exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(data_dir / "document_index.faiss"))
            
            # Save metadata
            with open(data_dir / "chunk_metadata.json", 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2)
                
            logger.info("Saved document index and metadata")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    async def load_existing(self) -> bool:
        """Load existing document index"""
        return self._load_index()