import trafilatura
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import re
import time
from urllib.parse import urljoin, urlparse
from document_processor import DocumentProcessor

class MOSDACWebScraper:
    def __init__(self):
        """Initialize the MOSDAC web scraper"""
        self.base_url = "https://www.mosdac.gov.in"
        self.document_processor = DocumentProcessor()
        self.scraped_urls = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_website_text_content(self, url: str) -> Optional[str]:
        """
        Extract main text content from a website URL using trafilatura
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text
            return None
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single URL and return structured content
        """
        if url in self.scraped_urls:
            return None
        
        try:
            # Extract main text content
            main_content = self.get_website_text_content(url)
            
            if not main_content:
                return None
            
            # Get additional metadata using BeautifulSoup
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc and hasattr(meta_desc, 'get') else ""
            
            # Extract keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            keywords = meta_keywords.get('content', '') if meta_keywords and hasattr(meta_keywords, 'get') else ""
            
            # Extract links to other pages
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    full_url = urljoin(url, href)
                    if self._is_valid_mosdac_url(full_url):
                        links.append({
                            'url': full_url,
                            'text': link.get_text().strip(),
                            'title': link.get('title', '')
                        })
            
            # Mark URL as scraped
            self.scraped_urls.add(url)
            
            content_data = {
                'url': url,
                'title': title_text,
                'description': description,
                'keywords': keywords,
                'text': main_content,
                'links': links[:20],  # Limit to first 20 links
                'scraped_at': time.time()
            }
            
            return content_data
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def scrape_mosdac_sections(self) -> List[Dict[str, Any]]:
        """
        Scrape key sections of the MOSDAC portal
        """
        key_urls = [
            f"{self.base_url}/",
            f"{self.base_url}/faq",
            f"{self.base_url}/data",
            f"{self.base_url}/services",
            f"{self.base_url}/about",
            f"{self.base_url}/help",
            f"{self.base_url}/products",
            f"{self.base_url}/missions",
            f"{self.base_url}/download"
        ]
        
        scraped_content = []
        
        for url in key_urls:
            try:
                content = self.scrape_url(url)
                if content:
                    scraped_content.append(content)
                
                # Be respectful with requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed to scrape {url}: {str(e)}")
                continue
        
        return scraped_content
    
    def discover_and_scrape_pages(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """
        Discover and scrape pages by following links from main pages
        """
        scraped_content = []
        urls_to_visit = [self.base_url]
        visited_count = 0
        
        while urls_to_visit and visited_count < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.scraped_urls:
                continue
            
            content = self.scrape_url(current_url)
            if content:
                scraped_content.append(content)
                
                # Add discovered links to queue
                for link in content.get('links', []):
                    link_url = link['url']
                    if (link_url not in self.scraped_urls and 
                        link_url not in urls_to_visit and
                        len(urls_to_visit) < max_pages * 2):
                        urls_to_visit.append(link_url)
            
            visited_count += 1
            time.sleep(1)  # Be respectful
        
        return scraped_content
    
    def scrape_documents(self, document_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape and process document files (PDF, DOCX, etc.)
        """
        processed_docs = []
        
        for url in document_urls:
            try:
                # Download the document
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine file type from URL or content-type
                content_type = response.headers.get('content-type', '')
                file_extension = self._get_file_extension(url, content_type)
                
                if file_extension in ['.pdf', '.docx', '.xlsx', '.txt']:
                    # Process document content
                    content = self.document_processor.process_document(
                        response.content, 
                        file_extension
                    )
                    
                    if content:
                        processed_docs.append({
                            'url': url,
                            'type': 'document',
                            'file_type': file_extension,
                            'title': self._extract_filename(url),
                            'text': content,
                            'scraped_at': time.time()
                        })
                
                time.sleep(2)  # Longer delay for documents
                
            except Exception as e:
                print(f"Error processing document {url}: {str(e)}")
                continue
        
        return processed_docs
    
    def _is_valid_mosdac_url(self, url: str) -> bool:
        """Check if URL is a valid MOSDAC portal URL"""
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == 'www.mosdac.gov.in' or 
                parsed.netloc == 'mosdac.gov.in'
            ) and not any(
                exclude in url.lower() 
                for exclude in ['javascript:', 'mailto:', '#', 'tel:']
            )
        except:
            return False
    
    def _get_file_extension(self, url: str, content_type: str) -> str:
        """Determine file extension from URL or content type"""
        # Try to get extension from URL
        parsed_url = urlparse(url)
        if '.' in parsed_url.path:
            return '.' + parsed_url.path.split('.')[-1].lower()
        
        # Map content types to extensions
        content_type_map = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'text/plain': '.txt'
        }
        
        return content_type_map.get(content_type, '.unknown')
    
    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1]
        return filename if filename else "Unknown Document"
    
    def get_scraping_summary(self) -> Dict[str, Any]:
        """Get summary of scraping activity"""
        return {
            'total_urls_scraped': len(self.scraped_urls),
            'scraped_urls': list(self.scraped_urls)
        }
