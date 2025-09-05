# Overview

The MOSDAC AI Help Bot is an intelligent virtual assistant designed for the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal. The system combines web scraping, natural language processing, knowledge graph construction, and conversational AI to provide users with accurate information retrieval from satellite data services, FAQs, documentation, and technical specifications. Built with a modular architecture, the bot can be deployed across similar web portals and provides contextual, relationship-based information discovery for geospatial data intelligence.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Single-page application providing chat interface and admin panel
- **Session Management**: Streamlit session state maintains chatbot instance, knowledge graph, chat history, and web scraper
- **Admin Panel**: Sidebar interface for knowledge graph initialization and portal scraping operations

## Backend Architecture
- **ChatBot Module**: Core conversational AI using OpenAI GPT-5 API with MOSDAC-specific system prompts
- **Knowledge Graph Engine**: NetworkX-based graph database storing entities, relationships, and semantic connections
- **NLP Processing Pipeline**: spaCy-powered text analysis with domain-specific pattern recognition for satellites, data products, and services
- **Web Scraping System**: Trafilatura and BeautifulSoup-based content extraction with rate limiting and session management

## Data Processing Components
- **Document Processor**: Multi-format document parser supporting PDF, DOCX, XLSX, and TXT files
- **Vector Store**: Sentence transformer-based semantic search using cosine similarity for document retrieval
- **Content Extraction**: Automated extraction of structured and unstructured web content including metadata, tables, and ARIA labels

## AI/ML Components
- **Semantic Search**: Vector embeddings using SentenceTransformer models for contextual information retrieval
- **Entity Recognition**: Custom NLP patterns for MOSDAC-specific entities (satellites, data products, technical terms)
- **Conversational Context**: Multi-turn conversation support with historical context preservation
- **Knowledge Graph Reasoning**: Relationship-based information discovery for complex queries

## Data Storage Strategy
- **In-Memory Graph**: NetworkX directed graph for real-time entity and relationship storage
- **Vector Embeddings**: In-memory vector store for semantic search capabilities
- **Session Persistence**: Streamlit session state for maintaining user context and chat history
- **Content Cache**: Scraped content caching to avoid redundant web requests

# External Dependencies

## AI/ML Services
- **OpenAI API**: GPT-5 model for conversational AI and query understanding
- **SentenceTransformers**: all-MiniLM-L6-v2 model for text embeddings and semantic search
- **spaCy**: en_core_web_sm model for natural language processing and entity extraction

## Web Technologies
- **Streamlit**: Web application framework for frontend interface
- **Trafilatura**: Content extraction from web pages
- **BeautifulSoup4**: HTML parsing and metadata extraction
- **Requests**: HTTP client for web scraping with session management

## Document Processing
- **PyPDF2**: PDF document text extraction
- **python-docx**: Microsoft Word document processing
- **openpyxl**: Excel spreadsheet processing

## Data Science Libraries
- **NetworkX**: Graph database and relationship modeling
- **NumPy**: Numerical computations for vector operations
- **scikit-learn**: Cosine similarity calculations for semantic search

## Target Integration
- **MOSDAC Portal**: Primary data source at www.mosdac.gov.in including FAQs, documentation, and satellite data services
- **Satellite Data Systems**: Integration with INSAT, OCEANSAT, CARTOSAT, and RESOURCESAT mission data
- **Geospatial Services**: Support for NetCDF, HDF, GeoTIFF formats and OGC web services