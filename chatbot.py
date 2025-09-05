import json
import os
from typing import Dict, List, Any
from openai import OpenAI
from knowledge_graph import KnowledgeGraph
from nlp_processor import NLPProcessor

class ChatBot:
    def __init__(self):
        """Initialize the chatbot with OpenAI integration"""
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        )
        self.nlp_processor = NLPProcessor()
        self.conversation_history = []
        
        # System prompt for MOSDAC assistance
        self.system_prompt = """
You are an intelligent assistant for the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal. 
Your role is to help users navigate and understand satellite data, services, and information available on the MOSDAC portal.

Key areas of expertise:
- Satellite missions (INSAT, OCEANSAT, CARTOSAT, RESOURCESAT, etc.)
- Ocean and meteorological data products (SST, Chlorophyll, Wind data, etc.)
- Data download procedures and portal services
- Technical specifications and metadata
- Geospatial data and visualization tools

Guidelines:
1. Provide accurate, helpful, and contextual responses
2. Use information from the knowledge graph when available
3. Be specific about data products, satellites, and procedures
4. If you don't have specific information, guide users to appropriate portal sections
5. Maintain a professional and helpful tone
6. Provide step-by-step instructions when needed
7. Include relevant technical details when appropriate

Always prioritize accuracy and helpfulness in your responses.
"""
    
    def get_response(self, user_query: str, knowledge_graph: KnowledgeGraph) -> str:
        """Generate a response to user query using knowledge graph context"""
        try:
            # Preprocess the query
            query_analysis = self.nlp_processor.preprocess_query(user_query)
            
            # Search knowledge graph for relevant information
            relevant_info = knowledge_graph.semantic_search(user_query, top_k=5)
            
            # Build context from knowledge graph
            context = self._build_context_from_kg(relevant_info, query_analysis)
            
            # Generate response using OpenAI
            response = self._generate_openai_response(user_query, context, query_analysis)
            
            # Update conversation history
            self._update_conversation_history(user_query, response)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
    
    def _build_context_from_kg(self, relevant_info: List[Dict], query_analysis: Dict) -> str:
        """Build context string from knowledge graph search results"""
        if not relevant_info:
            return "No specific information found in the knowledge base."
        
        context_parts = ["Relevant information from MOSDAC knowledge base:"]
        
        for i, info in enumerate(relevant_info[:3], 1):  # Limit to top 3 results
            entity = info.get('entity', 'Unknown')
            entity_type = info.get('type', 'Unknown')
            description = info.get('description', 'No description available')
            related_entities = info.get('related_entities', [])
            
            context_parts.append(f"\n{i}. {entity} ({entity_type})")
            context_parts.append(f"   Description: {description}")
            
            if related_entities:
                related_text = ", ".join([f"{rel[0]} ({rel[1]})" for rel in related_entities[:3]])
                context_parts.append(f"   Related to: {related_text}")
        
        # Add query analysis context
        if query_analysis.get('entities'):
            entities_text = ", ".join([ent['text'] for ent in query_analysis['entities'][:3]])
            context_parts.append(f"\nDetected entities in query: {entities_text}")
        
        if query_analysis.get('intent'):
            context_parts.append(f"Query intent: {query_analysis['intent']}")
        
        return "\n".join(context_parts)
    
    def _generate_openai_response(self, user_query: str, context: str, query_analysis: Dict) -> str:
        """Generate response using OpenAI GPT-5"""
        try:
            # Build conversation messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add recent conversation history for context
            for hist in self.conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": hist['user']})
                messages.append({"role": "assistant", "content": hist['assistant']})
            
            # Add current query with context
            user_message = f"""
Context from MOSDAC knowledge base:
{context}

User Query: {user_query}

Query Analysis:
- Intent: {query_analysis.get('intent', 'unknown')}
- Question Type: {query_analysis.get('question_type', 'unknown')}
- Keywords: {', '.join(query_analysis.get('keywords', []))}

Please provide a helpful and accurate response based on the context and your knowledge of MOSDAC portal.
"""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5"
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I'm sorry, but I'm having trouble generating a response right now. Error: {str(e)}"
    
    def _update_conversation_history(self, user_query: str, response: str):
        """Update conversation history for context"""
        self.conversation_history.append({
            'user': user_query,
            'assistant': response
        })
        
        # Keep only last 10 exchanges to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_intent_specific_response(self, intent: str, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Generate intent-specific responses"""
        intent_handlers = {
            'search_data': self._handle_data_search,
            'how_to': self._handle_how_to,
            'what_is': self._handle_what_is,
            'help': self._handle_help_request,
            'information': self._handle_information_request
        }
        
        handler = intent_handlers.get(intent, self._handle_general_query)
        return handler(entities, knowledge_graph)
    
    def _handle_data_search(self, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Handle data search queries"""
        if not entities:
            return """
To search for data on MOSDAC portal:
1. Visit the Data section
2. Use the search filters to narrow down by:
   - Satellite/Mission
   - Product type
   - Date range
   - Geographic region
3. Preview and download the required datasets

Popular data products include:
- Sea Surface Temperature (SST)
- Ocean Color/Chlorophyll
- Wind Speed and Direction
- Land Cover data
- Bathymetry data
"""
        
        # Search for specific entities mentioned
        entity_info = []
        for entity in entities[:3]:
            info = knowledge_graph.get_entity_info(entity['text'])
            if info:
                entity_info.append(info)
        
        if entity_info:
            response = "Based on your query, here's information about the requested data:\n\n"
            for info in entity_info:
                response += f"**{info['entity']}** ({info['type']})\n"
                response += f"- {info['attributes'].get('description', 'No description available')}\n"
                if info['related_entities']:
                    related = [rel[0] for rel in info['related_entities'][:3]]
                    response += f"- Related: {', '.join(related)}\n"
                response += "\n"
            return response
        
        return "I can help you find specific datasets. Please specify which satellite data or product you're looking for."
    
    def _handle_how_to(self, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Handle how-to queries"""
        return """
Here are common procedures for the MOSDAC portal:

**How to download data:**
1. Register/Login to your MOSDAC account
2. Navigate to the Data section
3. Use search filters to find your dataset
4. Select the data and add to cart
5. Proceed to download

**How to visualize data:**
1. Use the Visualization tools in the portal
2. Select your dataset and region of interest
3. Choose visualization parameters
4. Generate maps, plots, or animations

**How to access via API:**
1. Obtain API credentials from your profile
2. Use the documented API endpoints
3. Authenticate your requests
4. Retrieve data programmatically

Need help with a specific procedure? Please let me know!
"""
    
    def _handle_what_is(self, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Handle definition/explanation queries"""
        if entities:
            # Try to find information about the specific entity
            entity_text = entities[0]['text']
            info = knowledge_graph.get_entity_info(entity_text)
            
            if info:
                response = f"**{info['entity']}** is {info['attributes'].get('description', 'a component of the MOSDAC system')}."
                
                if info['related_entities']:
                    related = [f"{rel[0]} ({rel[1]})" for rel in info['related_entities'][:3]]
                    response += f"\n\nIt is related to: {', '.join(related)}"
                
                return response
        
        return """
MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) is India's premier facility for:

- **Satellite Data Management**: Archiving and distributing ocean and atmospheric data
- **Data Products**: Processing raw satellite data into useful scientific products
- **User Services**: Providing data access, visualization, and analysis tools
- **Research Support**: Supporting oceanographic and meteorological research

Key features include data from ISRO satellites like OCEANSAT, INSAT, CARTOSAT, and international missions.
"""
    
    def _handle_help_request(self, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Handle general help requests"""
        return """
I'm here to help you with MOSDAC portal! I can assist with:

🛰️ **Satellite Data**
- Information about satellite missions and instruments
- Available data products and their specifications
- Data quality and processing levels

📊 **Data Access**
- How to search and find datasets
- Download procedures and formats
- API access and programmatic retrieval

🗺️ **Visualization Tools**
- Creating maps and plots
- Animation and time-series tools
- Subsetting and analysis features

👤 **Account & Support**
- Registration and login issues
- User profile management
- Technical support contacts

What specific area would you like help with?
"""
    
    def _handle_information_request(self, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Handle general information requests"""
        return self._handle_what_is(entities, knowledge_graph)
    
    def _handle_general_query(self, entities: List[Dict], knowledge_graph: KnowledgeGraph) -> str:
        """Handle general queries"""
        return "I'm here to help you with MOSDAC portal information. Please feel free to ask about satellite data, download procedures, visualization tools, or any specific datasets you're looking for."
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            'total_exchanges': len(self.conversation_history),
            'recent_topics': [hist['user'][:50] + "..." for hist in self.conversation_history[-3:]],
            'conversation_length': sum(len(hist['user']) + len(hist['assistant']) for hist in self.conversation_history)
        }
