import re
from typing import List, Dict, Any, Tuple
import os

try:
    import spacy
except ImportError:
    spacy = None

class NLPProcessor:
    def __init__(self):
        """Initialize the NLP processor with spaCy"""
        if spacy is not None:
            try:
                # Try to load the English model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback: create a blank model if en_core_web_sm is not available
                print("Warning: en_core_web_sm model not found. Using blank model.")
                self.nlp = spacy.blank("en")
                # Add basic components
                if not self.nlp.has_pipe("tokenizer"):
                    self.nlp.add_pipe("tokenizer")
        else:
            print("Warning: spaCy not available. Using basic text processing.")
            self.nlp = None
        
        # Domain-specific patterns for MOSDAC
        self.mosdac_patterns = {
            'satellites': [
                'INSAT', 'OCEANSAT', 'CARTOSAT', 'RESOURCESAT', 'RISAT', 'SARAL',
                'SCATSAT', 'MEGHA-TROPIQUES', 'ASTROSAT'
            ],
            'data_products': [
                'SST', 'Sea Surface Temperature', 'Chlorophyll', 'Wind Speed',
                'Significant Wave Height', 'Bathymetry', 'Land Cover',
                'NDVI', 'Ocean Color', 'Altimetry'
            ],
            'services': [
                'Data Download', 'Visualization', 'API', 'Subsetting',
                'Time Series', 'Animation', 'Browse Products'
            ],
            'technical_terms': [
                'NetCDF', 'HDF', 'GeoTIFF', 'WMS', 'WCS', 'OGC',
                'Metadata', 'Projection', 'Coordinate System'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not text:
            return []
        
        entities = []
        
        try:
            # Process with spaCy if available
            if self.nlp is not None:
                doc = self.nlp(text)
                
                # Extract standard named entities
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8  # Default confidence for spaCy entities
                    })
            
            # Extract domain-specific entities
            domain_entities = self._extract_domain_entities(text)
            entities.extend(domain_entities)
            
        except Exception as e:
            print(f"Error in entity extraction: {str(e)}")
        
        return self._deduplicate_entities(entities)
    
    def _extract_domain_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract domain-specific entities using pattern matching"""
        entities = []
        text_upper = text.upper()
        
        for category, patterns in self.mosdac_patterns.items():
            for pattern in patterns:
                pattern_upper = pattern.upper()
                if pattern_upper in text_upper:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_upper.find(pattern_upper, start)
                        if pos == -1:
                            break
                        
                        entities.append({
                            'text': text[pos:pos+len(pattern)],
                            'label': category.upper(),
                            'start': pos,
                            'end': pos + len(pattern),
                            'confidence': 0.9
                        })
                        start = pos + 1
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on text and position"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['text'].lower(), entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships between entities"""
        if not text:
            return []
        
        relationships = []
        
        try:
            doc = self.nlp(text)
            
            # Simple pattern-based relationship extraction
            relationship_patterns = [
                (r'(\w+)\s+(provides|offers|contains|includes|generates)\s+(\w+)', 'provides'),
                (r'(\w+)\s+(is|are)\s+(part of|component of|used by)\s+(\w+)', 'part_of'),
                (r'(\w+)\s+(monitors|observes|measures)\s+(\w+)', 'monitors'),
                (r'(\w+)\s+(data|information)\s+(from|of)\s+(\w+)', 'source_of'),
                (r'(\w+)\s+(satellite|mission)\s+(carries|has)\s+(\w+)', 'carries')
            ]
            
            for pattern, relation_type in relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 3:
                        source = groups[0].strip()
                        target = groups[-1].strip()
                        
                        # Filter out common words
                        if not self._is_common_word(source) and not self._is_common_word(target):
                            relationships.append({
                                'source': source,
                                'target': target,
                                'relation': relation_type
                            })
            
            # Extract relationships using dependency parsing
            if hasattr(doc, 'sents'):
                dep_relationships = self._extract_dependency_relationships(doc)
                relationships.extend(dep_relationships)
            
        except Exception as e:
            print(f"Error in relationship extraction: {str(e)}")
        
        return relationships
    
    def _extract_dependency_relationships(self, doc) -> List[Dict[str, str]]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        try:
            for token in doc:
                # Look for subject-verb-object patterns
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    verb = token.head
                    subject = token.text
                    
                    # Find objects
                    for child in verb.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            obj = child.text
                            
                            if not self._is_common_word(subject) and not self._is_common_word(obj):
                                relationships.append({
                                    'source': subject,
                                    'target': obj,
                                    'relation': verb.lemma_
                                })
        except Exception as e:
            print(f"Error in dependency relationship extraction: {str(e)}")
        
        return relationships
    
    def _is_common_word(self, word: str) -> bool:
        """Check if a word is too common to be useful in relationships"""
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }
        return word.lower() in common_words or len(word) < 2
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []
        
        try:
            doc = self.nlp(text)
            
            # Collect candidate keywords
            keywords = []
            
            for token in doc:
                # Include nouns, proper nouns, and adjectives
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Count frequency and return most common
            from collections import Counter
            keyword_counts = Counter(keywords)
            
            return [word for word, count in keyword_counts.most_common(max_keywords)]
            
        except Exception as e:
            print(f"Error in keyword extraction: {str(e)}")
            return []
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess a user query for better understanding"""
        try:
            doc = self.nlp(query)
            
            # Extract query components
            entities = self.extract_entities(query)
            keywords = self.extract_keywords(query, max_keywords=5)
            
            # Classify query intent
            intent = self._classify_intent(query)
            
            # Extract question type
            question_type = self._get_question_type(query)
            
            return {
                'original_query': query,
                'entities': entities,
                'keywords': keywords,
                'intent': intent,
                'question_type': question_type,
                'tokens': [token.text for token in doc if not token.is_punct]
            }
            
        except Exception as e:
            print(f"Error in query preprocessing: {str(e)}")
            return {
                'original_query': query,
                'entities': [],
                'keywords': [],
                'intent': 'unknown',
                'question_type': 'unknown',
                'tokens': query.split()
            }
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of a user query"""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            'search_data': ['find', 'search', 'look for', 'get', 'download', 'access'],
            'how_to': ['how to', 'how do', 'how can', 'steps to'],
            'what_is': ['what is', 'what are', 'define', 'explain'],
            'where_is': ['where is', 'where can', 'location of'],
            'when_is': ['when is', 'when did', 'when will'],
            'help': ['help', 'assist', 'support', 'trouble', 'problem'],
            'information': ['information', 'details', 'about', 'describe']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return 'general'
    
    def _get_question_type(self, query: str) -> str:
        """Determine the type of question being asked"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith(('what', 'which')):
            return 'what'
        elif query_lower.startswith('how'):
            return 'how'
        elif query_lower.startswith('where'):
            return 'where'
        elif query_lower.startswith('when'):
            return 'when'
        elif query_lower.startswith('why'):
            return 'why'
        elif query_lower.startswith('who'):
            return 'who'
        elif '?' in query:
            return 'question'
        else:
            return 'statement'
