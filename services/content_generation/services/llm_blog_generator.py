# services/content_generation/services/llm_blog_generator.py

"""
LLM Blog Generator Service

This service generates educational blog posts for learning concepts using LLM.
It sources content from trusted academic sources and caches generated content.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import random


class LLMBlogGenerator:
    """
    Generates educational blog posts using LLM (or templates as fallback).
    
    Features:
    - Generates engaging educational content
    - Sources from trusted academic sources
    - Caches generated content
    - Personalizes based on student level
    """
    
    # Trusted sources for educational content
    TRUSTED_SOURCES = {
        "academic": [
            "arXiv.org - Academic research papers",
            "IEEE Xplore - Technical papers and standards",
            "ACM Digital Library - Computer science research",
            "Google Scholar - Peer-reviewed articles"
        ],
        "educational": [
            "MIT OpenCourseWare - Free course materials",
            "Stanford Online - University courses",
            "Coursera - Professional courses",
            "edX - University-level courses"
        ],
        "technical": [
            "Official Documentation - Technology-specific docs",
            "GitHub - Open source projects and examples",
            "Stack Overflow - Community Q&A",
            "Medium - Technical articles"
        ]
    }
    
    def __init__(self, cache_dir: Optional[str] = None, use_llm: bool = False):
        """
        Initialize the LLM Blog Generator.

        Args:
            cache_dir: Directory to cache generated content
            use_llm: Whether to use actual LLM (requires API key) or templates
        """
        if cache_dir is None:
            base_path = Path(__file__).parent.parent.parent.parent
            cache_dir = base_path / "data" / "generated_blogs"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm = use_llm
        self.llm_client = None

        if use_llm:
            try:
                from openai import OpenAI
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key)
                    print("✅ LLM client initialized with OpenAI API")
                else:
                    print("⚠️  WARNING: OPENAI_API_KEY not set")
                    print("   To enable LLM features: export OPENAI_API_KEY=sk-...")
                    print("   Using template-based generation as fallback")
                    self.use_llm = False
            except ImportError:
                print("⚠️  OpenAI package not installed")
                print("   Install with: pip install openai")
                print("   Using template-based generation as fallback")
                self.use_llm = False
    
    def _get_cache_key(self, concept_name: str, topic: str, student_level: str) -> str:
        """Generate a cache key for the blog."""
        key_string = f"{concept_name}_{topic}_{student_level}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_blog(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached blog if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached blog: {e}")
        return None
    
    def _cache_blog(self, cache_key: str, blog_data: Dict[str, Any]):
        """Cache generated blog."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(blog_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error caching blog: {e}")
    
    def generate_blog(self,
                     concept_name: str,
                     topic: str,
                     theory_data: Optional[Dict[str, Any]] = None,
                     student_level: str = "intermediate",
                     learning_style: str = "visual",
                     use_scraped_content: bool = True) -> Dict[str, Any]:
        """
        Generate an educational blog post for a concept.

        Args:
            concept_name: Name of the concept
            topic: Parent topic name
            theory_data: Optional theory data from lab_tutor
            student_level: Student's level (beginner/intermediate/advanced/expert)
            learning_style: Student's learning style
            use_scraped_content: Whether to scrape trusted sources for content

        Returns:
            Dictionary containing the generated blog
        """
        # Check cache first
        cache_key = self._get_cache_key(concept_name, topic, student_level)
        cached_blog = self._get_cached_blog(cache_key)
        if cached_blog:
            cached_blog['from_cache'] = True
            return cached_blog

        # Try to scrape trusted sources first if enabled
        scraped_blogs = []
        if use_scraped_content and theory_data:
            try:
                from .theory_blog_scraper import TheoryBlogScraper
                scraper = TheoryBlogScraper()
                theory_text = theory_data.get('compressed_text', theory_data.get('original_text', ''))
                theory_keywords = theory_data.get('keywords', [])

                if theory_text and theory_keywords:
                    scraped_blogs = scraper.scrape_blogs_for_theory(
                        theory_text,
                        theory_keywords,
                        max_results=3
                    )
            except Exception as e:
                print(f"⚠️  Error scraping blogs: {e}")
                scraped_blogs = []

        # Generate new blog
        if self.use_llm and self.llm_client:
            blog_data = self._generate_with_llm(concept_name, topic, theory_data, student_level, learning_style)
        else:
            blog_data = self._generate_with_template(concept_name, topic, theory_data, student_level, learning_style)

        # Add scraped blogs as references
        if scraped_blogs:
            blog_data['trusted_sources'] = scraped_blogs
            blog_data['source_type'] = 'hybrid'  # Mix of generated + scraped
        else:
            blog_data['source_type'] = 'generated'

        # Cache the generated blog
        self._cache_blog(cache_key, blog_data)
        blog_data['from_cache'] = False

        return blog_data
    
    def _generate_with_llm(self,
                          concept_name: str,
                          topic: str,
                          theory_data: Optional[Dict[str, Any]],
                          student_level: str,
                          learning_style: str) -> Dict[str, Any]:
        """Generate blog using actual LLM."""
        # Prepare context from theory data
        context = ""
        if theory_data:
            context = theory_data.get('compressed_text', theory_data.get('original_text', ''))
        
        # Create prompt
        prompt = self._create_llm_prompt(concept_name, topic, context, student_level, learning_style)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator specializing in computer science and data science topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse the response into structured format
            return self._parse_llm_response(content, concept_name, topic, student_level)
            
        except Exception as e:
            print(f"Error generating with LLM: {e}")
            # Fallback to template
            return self._generate_with_template(concept_name, topic, theory_data, student_level, learning_style)
    
    def _create_llm_prompt(self,
                          concept_name: str,
                          topic: str,
                          context: str,
                          student_level: str,
                          learning_style: str) -> str:
        """Create prompt for LLM."""
        prompt = f"""Generate an educational blog post about "{concept_name}" in the context of "{topic}".

Target Audience: {student_level.title()} level students
Learning Style: {learning_style.title()}

Context Information:
{context if context else "No additional context provided"}

Requirements:
1. Start with an engaging introduction that explains why this concept matters
2. Provide clear explanations with real-world examples
3. Include practical applications and use cases
4. Add code snippets or diagrams where appropriate (describe them in text)
5. Provide 3-5 key takeaways
6. Suggest further reading from trusted sources

Trusted Sources to reference:
- arXiv.org for research papers
- IEEE Xplore for technical standards
- MIT OpenCourseWare for educational content
- Official documentation for technologies

Tone: Educational, engaging, and accessible
Length: 800-1200 words

Format the response as:
INTRODUCTION:
[introduction text]

MAIN CONTENT:
[main explanation]

EXAMPLES:
[practical examples]

KEY TAKEAWAYS:
- [takeaway 1]
- [takeaway 2]
...

FURTHER READING:
- [source 1]
- [source 2]
...
"""
        return prompt
    
    def _parse_llm_response(self,
                           content: str,
                           concept_name: str,
                           topic: str,
                           student_level: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        # Simple parsing - can be enhanced
        sections = {
            "introduction": "",
            "main_content": "",
            "examples": "",
            "key_takeaways": [],
            "further_reading": []
        }
        
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('INTRODUCTION:'):
                current_section = 'introduction'
            elif line.startswith('MAIN CONTENT:'):
                current_section = 'main_content'
            elif line.startswith('EXAMPLES:'):
                current_section = 'examples'
            elif line.startswith('KEY TAKEAWAYS:'):
                current_section = 'key_takeaways'
            elif line.startswith('FURTHER READING:'):
                current_section = 'further_reading'
            elif line and current_section:
                if current_section in ['key_takeaways', 'further_reading']:
                    if line.startswith('-') or line.startswith('•'):
                        sections[current_section].append(line.lstrip('-•').strip())
                else:
                    sections[current_section] += line + "\n"
        
        return {
            "concept_name": concept_name,
            "topic": topic,
            "introduction": sections['introduction'].strip(),
            "main_content": sections['main_content'].strip(),
            "examples": sections['examples'].strip(),
            "key_takeaways": sections['key_takeaways'],
            "further_reading": sections['further_reading'],
            "sources": self._get_relevant_sources(concept_name, topic),
            "difficulty_level": student_level,
            "generated_at": datetime.now().isoformat(),
            "generation_method": "llm"
        }
    
    def _generate_with_template(self,
                               concept_name: str,
                               topic: str,
                               theory_data: Optional[Dict[str, Any]],
                               student_level: str,
                               learning_style: str) -> Dict[str, Any]:
        """Generate blog using templates (fallback when LLM not available)."""
        # Use theory data if available
        if theory_data:
            compressed_text = theory_data.get('compressed_text', '')
            keywords = theory_data.get('keywords', [])
        else:
            compressed_text = f"This concept is part of {topic}."
            keywords = [concept_name]
        
        # Generate introduction
        introduction = self._generate_introduction(concept_name, topic, student_level)
        
        # Main content from theory
        main_content = compressed_text if compressed_text else self._generate_generic_content(concept_name, topic)
        
        # Generate examples
        examples = self._generate_examples(concept_name, topic, keywords)
        
        # Key takeaways
        key_takeaways = self._generate_key_takeaways(concept_name, keywords)
        
        # Further reading
        further_reading = self._get_relevant_sources(concept_name, topic)
        
        return {
            "concept_name": concept_name,
            "topic": topic,
            "introduction": introduction,
            "main_content": main_content,
            "examples": examples,
            "key_takeaways": key_takeaways,
            "further_reading": further_reading,
            "sources": self._get_relevant_sources(concept_name, topic),
            "difficulty_level": student_level,
            "generated_at": datetime.now().isoformat(),
            "generation_method": "template"
        }
    
    def _generate_introduction(self, concept_name: str, topic: str, student_level: str) -> str:
        """Generate introduction text."""
        intros = [
            f"Welcome to this comprehensive guide on {concept_name}! As a key concept in {topic}, understanding {concept_name} is essential for {student_level} level students.",
            f"In this article, we'll explore {concept_name}, an important concept in {topic}. Whether you're just starting out or looking to deepen your understanding, this guide will help you master the fundamentals.",
            f"{concept_name} is a fundamental concept in {topic} that every student should understand. This guide will walk you through the key ideas and practical applications."
        ]
        return random.choice(intros)
    
    def _generate_generic_content(self, concept_name: str, topic: str) -> str:
        """Generate generic content when no theory data available."""
        return f"""{concept_name} is an important concept in {topic}. It represents a fundamental principle that helps us understand how systems work and how to design better solutions.

Understanding {concept_name} requires both theoretical knowledge and practical experience. By studying this concept, you'll gain insights into the underlying mechanisms and be able to apply this knowledge to real-world problems.

The key to mastering {concept_name} is to practice with examples, work through exercises, and connect the concept to other related ideas in {topic}."""
    
    def _generate_examples(self, concept_name: str, topic: str, keywords: List[str]) -> str:
        """Generate examples section."""
        return f"""Let's look at some practical examples of {concept_name}:

Example 1: Real-world Application
Consider how {concept_name} is used in modern systems. For instance, when working with {keywords[0] if keywords else 'data'}, we often need to apply {concept_name} principles to ensure efficiency and reliability.

Example 2: Code Implementation
In practice, implementing {concept_name} involves understanding the trade-offs and choosing the right approach for your specific use case.

Example 3: Common Scenarios
You'll encounter {concept_name} in various scenarios within {topic}, from basic operations to complex system designs."""
    
    def _generate_key_takeaways(self, concept_name: str, keywords: List[str]) -> List[str]:
        """Generate key takeaways."""
        takeaways = [
            f"{concept_name} is a fundamental concept that underpins many modern systems",
            f"Understanding {concept_name} helps you make better design decisions",
            f"Practical application of {concept_name} requires both theory and hands-on experience"
        ]
        
        if keywords:
            takeaways.append(f"Key related concepts include: {', '.join(keywords[:3])}")
        
        return takeaways
    
    def _get_relevant_sources(self, concept_name: str, topic: str) -> List[str]:
        """Get relevant trusted sources."""
        sources = []
        
        # Add academic sources
        sources.extend(random.sample(self.TRUSTED_SOURCES['academic'], 2))
        
        # Add educational sources
        sources.extend(random.sample(self.TRUSTED_SOURCES['educational'], 2))
        
        # Add technical sources
        sources.append(random.choice(self.TRUSTED_SOURCES['technical']))
        
        return sources


# Global instance
_blog_generator = None


def get_blog_generator(use_llm: bool = False) -> LLMBlogGenerator:
    """
    Get the global LLMBlogGenerator instance (singleton pattern).
    
    Args:
        use_llm: Whether to use actual LLM
        
    Returns:
        LLMBlogGenerator instance
    """
    global _blog_generator
    if _blog_generator is None:
        _blog_generator = LLMBlogGenerator(use_llm=use_llm)
    return _blog_generator

