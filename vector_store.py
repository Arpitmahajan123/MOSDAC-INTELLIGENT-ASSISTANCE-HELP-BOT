from typing import List, Tuple, Dict, Any
import json
import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector store with sentence transformer model"""
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                # Fallback to a basic model
                try:
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                except:
                    print("Warning: Could not load sentence transformer model. Some features may not work.")
                    self.model = None
        else:
            print("Warning: sentence_transformers not available. Vector search will be disabled.")
            self.model = None
        
        self.documents = {}  # doc_id -> document text
        self.embeddings = {}  # doc_id -> embedding vector
        self.metadata = {}   # doc_id -> metadata
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector store"""
        if not text or not self.model:
            return
        
        try:
            # Store document
            self.documents[doc_id] = text
            
            # Generate embedding
            embedding = self.model.encode(text)
            self.embeddings[doc_id] = embedding
            
            # Store metadata
            if metadata is not None:
                self.metadata[doc_id] = metadata
            
        except Exception as e:
            print(f"Error adding document {doc_id}: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using cosine similarity"""
        if not query or not self.model or not self.embeddings:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                if cosine_similarity is not None and np is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        doc_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((doc_id, float(similarity)))
                else:
                    # Fallback to simple text matching if sklearn not available
                    similarity = 0.5 if query.lower() in self.documents[doc_id].lower() else 0.1
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
    
    def get_document(self, doc_id: str) -> str:
        """Get document text by ID"""
        return self.documents.get(doc_id, "")
    
    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata by ID"""
        return self.metadata.get(doc_id, {})
    
    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find documents similar to a given document"""
        if doc_id not in self.documents:
            return []
        
        document_text = self.documents[doc_id]
        return self.search(document_text, top_k + 1)[1:]  # Exclude the document itself
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """Update an existing document"""
        if doc_id in self.documents:
            self.add_document(doc_id, text, metadata)
    
    def remove_document(self, doc_id: str):
        """Remove a document from the vector store"""
        self.documents.pop(doc_id, None)
        self.embeddings.pop(doc_id, None)
        self.metadata.pop(doc_id, None)
    
    def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the store"""
        return list(self.documents.keys())
    
    def get_document_count(self) -> int:
        """Get the total number of documents"""
        return len(self.documents)
    
    def save_to_file(self, filepath: str):
        """Save the vector store to a file"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
                'metadata': self.metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
    
    def load_from_file(self, filepath: str):
        """Load the vector store from a file"""
        try:
            if not os.path.exists(filepath):
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data.get('documents', {})
            self.metadata = data.get('metadata', {})
            
            # Convert embeddings back to numpy arrays
            embeddings_data = data.get('embeddings', {})
            if np is not None:
                self.embeddings = {
                    k: np.array(v) for k, v in embeddings_data.items()
                }
            else:
                self.embeddings = {}
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.documents.clear()
        self.embeddings.clear()
        self.metadata.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': len(next(iter(self.embeddings.values()))) if self.embeddings else 0,
            'model_name': self.model.get_sentence_embedding_dimension() if self.model else 'None'
        }
    
    def batch_add_documents(self, documents: List[Tuple[str, str, Dict[str, Any]]]):
        """Add multiple documents at once for better performance"""
        if not documents or not self.model:
            return
        
        try:
            # Prepare data
            doc_ids = []
            texts = []
            metadatas = []
            
            for doc_id, text, metadata in documents:
                if text:
                    doc_ids.append(doc_id)
                    texts.append(text)
                    metadatas.append(metadata or {})
            
            if not texts:
                return
            
            # Generate embeddings in batch
            embeddings = self.model.encode(texts)
            
            # Store all data
            for i, doc_id in enumerate(doc_ids):
                self.documents[doc_id] = texts[i]
                self.embeddings[doc_id] = embeddings[i]
                self.metadata[doc_id] = metadatas[i]
                
        except Exception as e:
            print(f"Error in batch add: {str(e)}")
    
    def semantic_search_with_filters(self, query: str, filters: Dict[str, Any] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search with metadata filters"""
        if not query or not self.model:
            return []
        
        # Get all search results first
        results = self.search(query, len(self.documents))
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for doc_id, score in results:
                doc_metadata = self.metadata.get(doc_id, {})
                
                # Check if document matches all filters
                matches = True
                for filter_key, filter_value in filters.items():
                    if filter_key not in doc_metadata or doc_metadata[filter_key] != filter_value:
                        matches = False
                        break
                
                if matches:
                    filtered_results.append((doc_id, score))
            
            results = filtered_results
        
        return results[:top_k]
