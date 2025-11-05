"""
Content Fetcher Agent - Fetches videos and reading materials from online sources
"""
import requests
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime


class ContentFetcherAgent:
    """
    Agent responsible for fetching educational content from trusted online sources.
    Handles both video content (YouTube) and reading materials (articles/blogs).
    """
    
    def __init__(self, youtube_api_key: Optional[str] = None):
        """
        Initialize the content fetcher agent.
        
        Args:
            youtube_api_key: Optional YouTube Data API key for fetching videos
        """
        self.youtube_api_key = youtube_api_key
        self.trusted_sources = [
            'medium.com',
            'dev.to',
            'towardsdatascience.com',
            'freecodecamp.org',
            'geeksforgeeks.org',
            'tutorialspoint.com',
            'w3schools.com',
            'mozilla.org',
            'stackoverflow.com',
            'github.com'
        ]
    
    def fetch_videos_for_topic(self, topic_name: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch educational videos for a topic from YouTube.
        
        Args:
            topic_name: Name of the topic
            max_results: Maximum number of videos to fetch
            
        Returns:
            List of video dictionaries with title, url, duration, views
        """
        videos = []
        
        if self.youtube_api_key:
            # Use YouTube Data API
            videos = self._fetch_from_youtube_api(topic_name, max_results)
        else:
            # Use template-based approach with curated educational channels
            videos = self._generate_template_videos(topic_name, max_results)
        
        return videos
    
    def _fetch_from_youtube_api(self, topic_name: str, max_results: int) -> List[Dict[str, Any]]:
        """Fetch videos using YouTube Data API"""
        videos = []
        
        try:
            # Search for educational videos
            search_url = "https://www.googleapis.com/youtube/v3/search"
            search_params = {
                'part': 'snippet',
                'q': f"{topic_name} tutorial educational",
                'type': 'video',
                'maxResults': max_results * 2,  # Get more to filter
                'order': 'relevance',
                'videoDuration': 'medium',  # 4-20 minutes
                'key': self.youtube_api_key
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get video details
            video_ids = [item['id']['videoId'] for item in data.get('items', [])]
            
            if video_ids:
                details_url = "https://www.googleapis.com/youtube/v3/videos"
                details_params = {
                    'part': 'contentDetails,statistics,snippet',
                    'id': ','.join(video_ids),
                    'key': self.youtube_api_key
                }
                
                details_response = requests.get(details_url, params=details_params, timeout=10)
                details_response.raise_for_status()
                details_data = details_response.json()
                
                for item in details_data.get('items', [])[:max_results]:
                    duration_str = item['contentDetails']['duration']
                    duration_minutes = self._parse_duration(duration_str)
                    
                    videos.append({
                        'title': item['snippet']['title'],
                        'url': f"https://www.youtube.com/embed/{item['id']}",
                        'duration': duration_minutes,
                        'views': int(item['statistics'].get('viewCount', 0)),
                        'channel': item['snippet']['channelTitle'],
                        'thumbnail': item['snippet']['thumbnails']['high']['url']
                    })
        
        except Exception as e:
            print(f"Error fetching from YouTube API: {e}")
            # Fallback to template videos
            videos = self._generate_template_videos(topic_name, max_results)
        
        return videos
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to minutes"""
        # PT15M33S -> 15 minutes
        match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            return hours * 60 + minutes
        return 0
    
    def _generate_template_videos(self, topic_name: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate template videos with educational channel URLs"""
        # Curated educational channels
        channels = [
            {'name': 'freeCodeCamp.org', 'id': 'UC8butISFwT-Wl7EV0hUK0BQ'},
            {'name': 'Traversy Media', 'id': 'UC29ju8bIPH5as8OGnQzwJyA'},
            {'name': 'Programming with Mosh', 'id': 'UCWv7vMbMWH4-V0ZXdmDpPBA'},
            {'name': 'Academind', 'id': 'UCSJbGtTlrDami-tDGPUV9-w'},
            {'name': 'The Net Ninja', 'id': 'UCW5YeuERMmlnqo4oq8vwUpg'}
        ]
        
        videos = []
        for i in range(min(max_results, 3)):
            channel = channels[i % len(channels)]
            videos.append({
                'title': f"{topic_name} - Complete Tutorial (Part {i+1})",
                'url': f"https://www.youtube.com/embed/dQw4w9WgXcQ",  # Placeholder
                'duration': 15 + (i * 5),
                'views': 500000 + (i * 100000),
                'channel': channel['name'],
                'thumbnail': f"https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"
            })
        
        return videos
    
    def fetch_reading_materials_for_concept(self, concept_name: str, topic_name: str, 
                                           max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch reading materials/articles for a concept from trusted sources.
        
        Args:
            concept_name: Name of the concept
            topic_name: Parent topic name
            max_results: Maximum number of articles to fetch
            
        Returns:
            List of reading material dictionaries
        """
        readings = []
        
        # Try to fetch from multiple sources
        readings.extend(self._fetch_from_medium(concept_name, topic_name, max_results=1))
        readings.extend(self._fetch_from_dev_to(concept_name, topic_name, max_results=1))
        readings.extend(self._generate_template_readings(concept_name, topic_name, max_results=1))
        
        return readings[:max_results]
    
    def _fetch_from_medium(self, concept_name: str, topic_name: str, max_results: int) -> List[Dict[str, Any]]:
        """Fetch articles from Medium (template-based)"""
        # Medium doesn't have a public API, so we use template URLs
        readings = []
        
        search_query = f"{concept_name} {topic_name}".replace(' ', '-').lower()
        
        readings.append({
            'title': f"Understanding {concept_name}: A Comprehensive Guide",
            'url': f"https://medium.com/search?q={search_query}",
            'source': 'Medium',
            'description': f"Deep dive into {concept_name} with practical examples and best practices.",
            'author': 'Tech Community',
            'read_time': 8,
            'concept_name': concept_name
        })
        
        return readings[:max_results]
    
    def _fetch_from_dev_to(self, concept_name: str, topic_name: str, max_results: int) -> List[Dict[str, Any]]:
        """Fetch articles from DEV.to"""
        readings = []
        
        try:
            # DEV.to has a public API
            url = "https://dev.to/api/articles"
            params = {
                'tag': concept_name.lower().replace(' ', ''),
                'per_page': max_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()
            
            for article in articles[:max_results]:
                readings.append({
                    'title': article['title'],
                    'url': article['url'],
                    'source': 'DEV.to',
                    'description': article.get('description', f"Learn about {concept_name}"),
                    'author': article['user']['name'],
                    'read_time': article.get('reading_time_minutes', 5),
                    'concept_name': concept_name
                })
        
        except Exception as e:
            print(f"Error fetching from DEV.to: {e}")
            # Return template reading
            readings.append({
                'title': f"{concept_name} Explained",
                'url': f"https://dev.to/search?q={concept_name.replace(' ', '%20')}",
                'source': 'DEV.to',
                'description': f"Community articles about {concept_name}",
                'author': 'DEV Community',
                'read_time': 6,
                'concept_name': concept_name
            })
        
        return readings
    
    def _generate_template_readings(self, concept_name: str, topic_name: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate template reading materials from trusted sources"""
        readings = []
        
        sources = [
            {
                'name': 'GeeksforGeeks',
                'url_template': 'https://www.geeksforgeeks.org/{query}/',
                'description': 'Comprehensive tutorial with examples and practice problems'
            },
            {
                'name': 'TutorialsPoint',
                'url_template': 'https://www.tutorialspoint.com/{query}.htm',
                'description': 'Step-by-step guide with code examples'
            },
            {
                'name': 'W3Schools',
                'url_template': 'https://www.w3schools.com/{query}.asp',
                'description': 'Interactive tutorial with try-it-yourself examples'
            }
        ]
        
        for i, source in enumerate(sources[:max_results]):
            query = concept_name.lower().replace(' ', '-')
            readings.append({
                'title': f"{concept_name} - {source['name']} Tutorial",
                'url': source['url_template'].format(query=query),
                'source': source['name'],
                'description': source['description'],
                'author': source['name'],
                'read_time': 10,
                'concept_name': concept_name
            })
        
        return readings
    
    def generate_personalized_blog(self, concept_name: str, concept_description: str,
                                   theory_text: str, student_hobbies: List[str],
                                   student_interests: List[str]) -> Dict[str, Any]:
        """
        Generate a personalized blog post for a concept based on student hobbies/interests.
        
        Args:
            concept_name: Name of the concept
            concept_description: Description of the concept
            theory_text: Theory text from knowledge graph
            student_hobbies: List of student hobbies
            student_interests: List of student interests
            
        Returns:
            Dictionary with blog content
        """
        # Create personalized examples based on hobbies
        hobby_examples = self._create_hobby_examples(concept_name, student_hobbies)
        
        blog_content = {
            'title': f"Understanding {concept_name}",
            'concept_name': concept_name,
            'introduction': concept_description or f"Let's explore {concept_name} in detail.",
            'theory': theory_text or f"Core concepts of {concept_name}",
            'personalized_examples': hobby_examples,
            'summary': f"You've learned about {concept_name} with examples from your interests!",
            'generated_at': datetime.now().isoformat()
        }
        
        return blog_content
    
    def _create_hobby_examples(self, concept_name: str, hobbies: List[str]) -> List[Dict[str, str]]:
        """Create examples based on student hobbies"""
        examples = []
        
        hobby_contexts = {
            'gaming': 'video game development',
            'sports': 'sports analytics',
            'music': 'music streaming services',
            'reading': 'digital library systems',
            'photography': 'photo management apps',
            'cooking': 'recipe recommendation systems',
            'technology': 'tech applications',
            'art': 'digital art platforms'
        }
        
        for hobby in hobbies[:2]:  # Use top 2 hobbies
            context = hobby_contexts.get(hobby.lower(), 'real-world applications')
            examples.append({
                'hobby': hobby,
                'example': f"In {context}, {concept_name} is used to..."
            })
        
        return examples

