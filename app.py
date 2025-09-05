import streamlit as st
import os
from chatbot import ChatBot
from knowledge_graph import KnowledgeGraph
from web_scraper import MOSDACWebScraper

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ChatBot()
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = KnowledgeGraph()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'scraper' not in st.session_state:
    st.session_state.scraper = MOSDACWebScraper()

st.set_page_config(
    page_title="MOSDAC AI Help Bot",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ MOSDAC AI Help Bot")
st.markdown("Intelligent assistance for MOSDAC portal information retrieval")

# Sidebar for admin functions
with st.sidebar:
    st.header("Admin Panel")
    
    # Knowledge Graph Management
    st.subheader("Knowledge Graph")
    
    if st.button("Initialize Knowledge Graph"):
        with st.spinner("Initializing knowledge graph..."):
            try:
                st.session_state.knowledge_graph.initialize_base_entities()
                st.success("Knowledge graph initialized!")
            except Exception as e:
                st.error(f"Error initializing knowledge graph: {str(e)}")
    
    if st.button("Scrape MOSDAC Portal"):
        with st.spinner("Scraping MOSDAC portal content..."):
            try:
                # Scrape main portal pages
                urls_to_scrape = [
                    "https://www.mosdac.gov.in/",
                    "https://www.mosdac.gov.in/faq",
                    "https://www.mosdac.gov.in/data",
                    "https://www.mosdac.gov.in/services"
                ]
                
                scraped_data = []
                for url in urls_to_scrape:
                    try:
                        content = st.session_state.scraper.scrape_url(url)
                        if content:
                            scraped_data.append(content)
                    except Exception as e:
                        st.warning(f"Failed to scrape {url}: {str(e)}")
                
                # Process scraped data into knowledge graph
                if scraped_data:
                    for data in scraped_data:
                        st.session_state.knowledge_graph.add_content(data)
                    
                    st.success(f"Successfully scraped and processed {len(scraped_data)} pages!")
                else:
                    st.warning("No content was successfully scraped")
                    
            except Exception as e:
                st.error(f"Error during scraping: {str(e)}")
    
    # Display knowledge graph stats
    if hasattr(st.session_state.knowledge_graph, 'graph'):
        st.metric("Entities", st.session_state.knowledge_graph.get_entity_count())
        st.metric("Relationships", st.session_state.knowledge_graph.get_relationship_count())

# Main chat interface
st.header("Chat with MOSDAC Assistant")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about MOSDAC portal..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    st.chat_message("user").write(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chatbot.get_response(
                    prompt, 
                    st.session_state.knowledge_graph
                )
                st.write(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer with information
st.markdown("---")
st.markdown("""
**Features:**
- 🔍 Semantic search across MOSDAC portal content
- 🧠 Knowledge graph-based information retrieval
- 🗣️ Natural language query understanding
- 📊 Geospatial data intelligence support
- 🔄 Real-time response generation

**Data Sources:** MOSDAC portal content including FAQs, documentation, product catalogs, and satellite mission details.
""")
