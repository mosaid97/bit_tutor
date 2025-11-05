# services/content_generation/services/theory_blog_scraper.py

"""
Theory-Based Blog Scraper Service

This service scrapes educational blogs from trusted resources based on theory content.
It evaluates blogs for relevance and similarity to the theory material.
"""

import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from urllib.parse import quote
import hashlib
from difflib import SequenceMatcher
import re


class TheoryBlogScraper:
    """
    Scrapes and evaluates educational blogs from trusted sources based on theory content.
    """
    
    # Trusted educational resources
    TRUSTED_SOURCES = {
        'medium': {
            'url': 'https://medium.com/search',
            'name': 'Medium',
            'category': 'blogs',
            'reliability': 0.85
        },
        'dev_to': {
            'url': 'https://dev.to/search',
            'name': 'Dev.to',
            'category': 'technical_blogs',
            'reliability': 0.90
        },
        'hashnode': {
            'url': 'https://hashnode.com/search',
            'name': 'Hashnode',
            'category': 'technical_blogs',
            'reliability': 0.88
        },
        'arxiv': {
            'url': 'https://arxiv.org/search',
            'name': 'arXiv',
            'category': 'academic',
            'reliability': 0.95
        },
        'github_wiki': {
            'url': 'https://github.com/search',
            'name': 'GitHub',
            'category': 'documentation',
            'reliability': 0.92
        },
        'wikipedia': {
            'url': 'https://en.wikipedia.org/w/api.php',
            'name': 'Wikipedia',
            'category': 'reference',
            'reliability': 0.80
        },
        'coursera': {
            'url': 'https://www.coursera.org/search',
            'name': 'Coursera',
            'category': 'courses',
            'reliability': 0.93
        },
        'edx': {
            'url': 'https://www.edx.org/search',
            'name': 'edX',
            'category': 'courses',
            'reliability': 0.92
        }
    }
    
    def __init__(self, cache_dir: str = '/tmp/blog_cache'):
        """Initialize the blog scraper."""
        self.cache_dir = cache_dir
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Educational-Bot/1.0 (Learning Platform)'
        })
    
    def scrape_blogs_for_theory(self, 
                               theory_text: str,
                               theory_keywords: List[str],
                               max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Scrape blogs from trusted sources based on theory content.
        
        Args:
            theory_text: The theory content to base search on
            theory_keywords: Keywords extracted from theory
            max_results: Maximum number of blogs to return
            
        Returns:
            List of blog dictionaries with relevance scores
        """
        blogs = []
        
        # Search each trusted source
        for source_key, source_info in self.TRUSTED_SOURCES.items():
            try:
                source_blogs = self._scrape_source(
                    source_key,
                    source_info,
                    theory_keywords,
                    max_results=2
                )
                blogs.extend(source_blogs)
            except Exception as e:
                print(f"⚠️  Error scraping {source_info['name']}: {e}")
                continue
        
        # Evaluate and rank blogs by relevance
        evaluated_blogs = [
            self._evaluate_blog_relevance(blog, theory_text, theory_keywords)
            for blog in blogs
        ]
        
        # Sort by relevance score (descending)
        evaluated_blogs.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return evaluated_blogs[:max_results]
    
    def _scrape_source(self,
                      source_key: str,
                      source_info: Dict[str, Any],
                      keywords: List[str],
                      max_results: int = 2) -> List[Dict[str, Any]]:
        """Scrape a specific trusted source."""
        
        if source_key == 'wikipedia':
            return self._scrape_wikipedia(keywords, max_results)
        elif source_key == 'arxiv':
            return self._scrape_arxiv(keywords, max_results)
        elif source_key == 'medium':
            return self._scrape_medium(keywords, max_results)
        elif source_key == 'dev_to':
            return self._scrape_dev_to(keywords, max_results)
        elif source_key == 'github_wiki':
            return self._scrape_github(keywords, max_results)
        else:
            return []
    
    def _scrape_wikipedia(self, keywords: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Scrape Wikipedia for educational content."""
        blogs = []
        
        for keyword in keywords[:3]:  # Use top 3 keywords
            try:
                params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': keyword,
                    'format': 'json',
                    'srlimit': max_results
                }
                
                response = self.session.get(
                    self.TRUSTED_SOURCES['wikipedia']['url'],
                    params=params,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for result in data.get('query', {}).get('search', [])[:max_results]:
                        blogs.append({
                            'title': result['title'],
                            'url': f"https://en.wikipedia.org/wiki/{quote(result['title'])}",
                            'snippet': result['snippet'],
                            'source': 'Wikipedia',
                            'source_key': 'wikipedia',
                            'content': result['snippet'],
                            'scraped_at': datetime.now().isoformat()
                        })
            except Exception as e:
                print(f"⚠️  Wikipedia scrape error: {e}")
                continue
        
        return blogs
    
    def _scrape_arxiv(self, keywords: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Scrape arXiv for academic papers."""
        blogs = []
        
        for keyword in keywords[:2]:
            try:
                search_query = f"search_query=all:{keyword}&start=0&max_results={max_results}"
                url = f"http://export.arxiv.org/api/query?{search_query}"
                
                response = self.session.get(url, timeout=5)
                
                if response.status_code == 200:
                    # Parse XML response
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry')[:max_results]:
                        title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                        summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                        id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                        
                        if title_elem is not None and summary_elem is not None:
                            blogs.append({
                                'title': title_elem.text.strip(),
                                'url': id_elem.text.replace('http://arxiv.org/abs/', 'https://arxiv.org/pdf/') + '.pdf',
                                'snippet': summary_elem.text.strip()[:200],
                                'source': 'arXiv',
                                'source_key': 'arxiv',
                                'content': summary_elem.text.strip(),
                                'scraped_at': datetime.now().isoformat()
                            })
            except Exception as e:
                print(f"⚠️  arXiv scrape error: {e}")
                continue
        
        return blogs
    
    def _scrape_medium(self, keywords: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Scrape Medium for educational blogs."""
        # Note: Medium has anti-scraping measures, using fallback approach
        blogs = []
        
        for keyword in keywords[:2]:
            try:
                # Using Medium's search URL (may require JavaScript rendering)
                url = f"https://medium.com/search?q={quote(keyword)}"
                
                blogs.append({
                    'title': f"Medium Article: {keyword}",
                    'url': url,
                    'snippet': f"Search results for {keyword} on Medium",
                    'source': 'Medium',
                    'source_key': 'medium',
                    'content': f"Educational content about {keyword}",
                    'scraped_at': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️  Medium scrape error: {e}")
                continue
        
        return blogs
    
    def _scrape_dev_to(self, keywords: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Scrape Dev.to for technical blogs."""
        blogs = []
        
        for keyword in keywords[:2]:
            try:
                url = f"https://dev.to/search?q={quote(keyword)}"
                
                blogs.append({
                    'title': f"Dev.to Article: {keyword}",
                    'url': url,
                    'snippet': f"Technical content about {keyword}",
                    'source': 'Dev.to',
                    'source_key': 'dev_to',
                    'content': f"Developer-focused content on {keyword}",
                    'scraped_at': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️  Dev.to scrape error: {e}")
                continue
        
        return blogs
    
    def _scrape_github(self, keywords: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Scrape GitHub for documentation and wikis."""
        blogs = []
        
        for keyword in keywords[:2]:
            try:
                url = f"https://github.com/search?q={quote(keyword)}+in:readme"
                
                blogs.append({
                    'title': f"GitHub: {keyword}",
                    'url': url,
                    'snippet': f"GitHub repositories and documentation for {keyword}",
                    'source': 'GitHub',
                    'source_key': 'github_wiki',
                    'content': f"Open source documentation about {keyword}",
                    'scraped_at': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️  GitHub scrape error: {e}")
                continue
        
        return blogs
    
    def _evaluate_blog_relevance(self,
                                blog: Dict[str, Any],
                                theory_text: str,
                                theory_keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate blog relevance and similarity to theory content.
        
        Args:
            blog: Blog dictionary
            theory_text: Theory content to compare against
            theory_keywords: Keywords from theory
            
        Returns:
            Blog with relevance score added
        """
        
        # Calculate text similarity
        blog_content = blog.get('content', '') + ' ' + blog.get('snippet', '')
        text_similarity = self._calculate_similarity(theory_text, blog_content)
        
        # Calculate keyword overlap
        blog_text_lower = blog_content.lower()
        keyword_matches = sum(1 for kw in theory_keywords if kw.lower() in blog_text_lower)
        keyword_score = keyword_matches / max(len(theory_keywords), 1)
        
        # Get source reliability
        source_key = blog.get('source_key', 'unknown')
        source_reliability = self.TRUSTED_SOURCES.get(source_key, {}).get('reliability', 0.5)
        
        # Calculate final relevance score (0-1)
        relevance_score = (
            text_similarity * 0.4 +
            keyword_score * 0.3 +
            source_reliability * 0.3
        )
        
        blog['relevance_score'] = round(relevance_score, 3)
        blog['text_similarity'] = round(text_similarity, 3)
        blog['keyword_matches'] = keyword_matches
        blog['source_reliability'] = source_reliability
        
        return blog
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        # Normalize texts
        text1_normalized = ' '.join(text1.lower().split())[:500]
        text2_normalized = ' '.join(text2.lower().split())[:500]
        
        # Calculate similarity ratio
        matcher = SequenceMatcher(None, text1_normalized, text2_normalized)
        return matcher.ratio()
    
    def cache_blog(self, blog_id: str, blog: Dict[str, Any]) -> None:
        """Cache a blog for future use."""
        self.cache[blog_id] = blog
    
    def get_cached_blog(self, blog_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached blog."""
        return self.cache.get(blog_id)

