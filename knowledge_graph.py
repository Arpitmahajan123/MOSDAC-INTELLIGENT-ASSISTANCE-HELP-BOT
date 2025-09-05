import networkx as nx
import json
import os
from typing import Dict, List, Any, Tuple
from nlp_processor import NLPProcessor
from vector_store import VectorStore

class KnowledgeGraph:
    def __init__(self):
        """Initialize the knowledge graph with NetworkX"""
        self.graph = nx.DiGraph()
        self.nlp_processor = NLPProcessor()
        self.vector_store = VectorStore()
        self.entity_types = {
            'SATELLITE': 'satellite missions and spacecraft',
            'PRODUCT': 'data products and datasets',
            'SERVICE': 'portal services and tools',
            'ORGANIZATION': 'organizations and institutions',
            'LOCATION': 'geographical locations',
            'DATE': 'dates and temporal information',
            'TECHNICAL': 'technical specifications and parameters'
        }
        
    def initialize_base_entities(self):
        """Initialize the knowledge graph with base MOSDAC entities"""
        base_entities = [
            # Satellites and Missions
            ('ISRO', 'ORGANIZATION', {'description': 'Indian Space Research Organisation'}),
            ('MOSDAC', 'ORGANIZATION', {'description': 'Meteorological and Oceanographic Satellite Data Archival Centre'}),
            ('INSAT', 'SATELLITE', {'description': 'Indian National Satellite System'}),
            ('OCEANSAT', 'SATELLITE', {'description': 'Ocean observation satellite series'}),
            ('CARTOSAT', 'SATELLITE', {'description': 'Earth observation satellite for cartographic applications'}),
            ('RESOURCESAT', 'SATELLITE', {'description': 'Earth observation satellite for resource monitoring'}),
            
            # Data Products
            ('SST', 'PRODUCT', {'description': 'Sea Surface Temperature data'}),
            ('Chlorophyll', 'PRODUCT', {'description': 'Ocean chlorophyll concentration data'}),
            ('Wind_Data', 'PRODUCT', {'description': 'Ocean wind speed and direction data'}),
            ('Land_Cover', 'PRODUCT', {'description': 'Land cover classification data'}),
            ('Bathymetry', 'PRODUCT', {'description': 'Ocean depth measurements'}),
            
            # Services
            ('Data_Download', 'SERVICE', {'description': 'Portal data download service'}),
            ('Visualization', 'SERVICE', {'description': 'Data visualization tools'}),
            ('API_Access', 'SERVICE', {'description': 'Programmatic data access'}),
            ('User_Registration', 'SERVICE', {'description': 'User account management'})
        ]
        
        # Add base entities to graph
        for entity_id, entity_type, attributes in base_entities:
            self.add_entity(entity_id, entity_type, attributes)
        
        # Add some basic relationships
        relationships = [
            ('ISRO', 'operates', 'MOSDAC'),
            ('MOSDAC', 'provides', 'Data_Download'),
            ('MOSDAC', 'provides', 'Visualization'),
            ('MOSDAC', 'provides', 'API_Access'),
            ('INSAT', 'generates', 'SST'),
            ('OCEANSAT', 'generates', 'Chlorophyll'),
            ('OCEANSAT', 'generates', 'Wind_Data'),
            ('CARTOSAT', 'generates', 'Land_Cover'),
            ('RESOURCESAT', 'generates', 'Land_Cover')
        ]
        
        for source, relation, target in relationships:
            self.add_relationship(source, target, relation)
    
    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, Any] = None):
        """Add an entity to the knowledge graph"""
        if attributes is None:
            attributes = {}
        
        self.graph.add_node(
            entity_id,
            type=entity_type,
            **attributes
        )
        
        # Add to vector store for semantic search
        entity_text = f"{entity_id} {entity_type} {attributes.get('description', '')}"
        self.vector_store.add_document(entity_id, entity_text)
    
    def add_relationship(self, source: str, target: str, relation: str, attributes: Dict[str, Any] = None):
        """Add a relationship between entities"""
        if attributes is None:
            attributes = {}
        
        self.graph.add_edge(source, target, relation=relation, **attributes)
    
    def add_content(self, content_data: Dict[str, Any]):
        """Process and add content to the knowledge graph"""
        try:
            # Extract entities from content
            entities = self.nlp_processor.extract_entities(content_data.get('text', ''))
            
            # Add entities to graph
            for entity in entities:
                entity_id = entity['text'].replace(' ', '_')
                entity_type = self._map_spacy_label_to_type(entity['label'])
                
                self.add_entity(
                    entity_id,
                    entity_type,
                    {
                        'description': f"Entity from {content_data.get('url', 'unknown source')}",
                        'source_url': content_data.get('url', ''),
                        'confidence': entity.get('confidence', 0.0)
                    }
                )
            
            # Extract relationships
            relationships = self.nlp_processor.extract_relationships(content_data.get('text', ''))
            for rel in relationships:
                self.add_relationship(
                    rel['source'].replace(' ', '_'),
                    rel['target'].replace(' ', '_'),
                    rel['relation']
                )
                
        except Exception as e:
            print(f"Error processing content: {str(e)}")
    
    def _map_spacy_label_to_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'DATE': 'DATE',
            'PERSON': 'ORGANIZATION',
            'PRODUCT': 'PRODUCT',
            'WORK_OF_ART': 'PRODUCT',
            'EVENT': 'SERVICE',
            'FAC': 'SERVICE'
        }
        return mapping.get(spacy_label, 'TECHNICAL')
    
    def find_related_entities(self, entity: str, max_depth: int = 2) -> List[Tuple[str, str]]:
        """Find entities related to the given entity"""
        if entity not in self.graph:
            return []
        
        related = []
        
        # Direct relationships
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            relation = edge_data.get('relation', 'related_to')
            related.append((neighbor, relation))
        
        # Reverse relationships
        for predecessor in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(predecessor, entity)
            relation = edge_data.get('relation', 'related_to')
            related.append((predecessor, f"inverse_{relation}"))
        
        return related
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on the knowledge graph"""
        # Get similar documents from vector store
        similar_docs = self.vector_store.search(query, top_k)
        
        results = []
        for doc_id, score in similar_docs:
            if doc_id in self.graph:
                node_data = self.graph.nodes[doc_id]
                results.append({
                    'entity': doc_id,
                    'type': node_data.get('type', 'UNKNOWN'),
                    'description': node_data.get('description', ''),
                    'score': score,
                    'related_entities': self.find_related_entities(doc_id)
                })
        
        return results
    
    def get_entity_info(self, entity: str) -> Dict[str, Any]:
        """Get detailed information about an entity"""
        if entity not in self.graph:
            return {}
        
        node_data = self.graph.nodes[entity]
        related_entities = self.find_related_entities(entity)
        
        return {
            'entity': entity,
            'type': node_data.get('type', 'UNKNOWN'),
            'attributes': dict(node_data),
            'related_entities': related_entities,
            'neighbors_count': len(list(self.graph.neighbors(entity)))
        }
    
    def get_entity_count(self) -> int:
        """Get the total number of entities in the graph"""
        return self.graph.number_of_nodes()
    
    def get_relationship_count(self) -> int:
        """Get the total number of relationships in the graph"""
        return self.graph.number_of_edges()
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export the knowledge graph data"""
        return {
            'nodes': [
                {
                    'id': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    **self.graph.edges[edge]
                }
                for edge in self.graph.edges()
            ]
        }
