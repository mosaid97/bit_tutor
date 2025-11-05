# services/content_generation/services/blog_relevance_evaluator.py

"""
Blog Relevance Evaluator Service

Evaluates scraped blogs for relevance, similarity, and quality based on theory content.
Uses multiple metrics to ensure educational value.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import re
from datetime import datetime


@dataclass
class RelevanceMetrics:
    """Metrics for blog relevance evaluation."""
    text_similarity: float  # 0-1: How similar is the blog to theory
    keyword_coverage: float  # 0-1: How many theory keywords are covered
    source_credibility: float  # 0-1: How credible is the source
    content_depth: float  # 0-1: How detailed is the content
    educational_value: float  # 0-1: Overall educational value
    overall_score: float  # 0-1: Final relevance score


class BlogRelevanceEvaluator:
    """
    Evaluates blogs for relevance and similarity to theory content.
    """
    
    # Credibility scores for different sources
    SOURCE_CREDIBILITY = {
        'arxiv': 0.95,
        'wikipedia': 0.85,
        'github_wiki': 0.90,
        'dev_to': 0.80,
        'medium': 0.75,
        'coursera': 0.92,
        'edx': 0.91,
        'hashnode': 0.82,
        'official_docs': 0.98,
        'academic_journal': 0.96,
        'unknown': 0.50
    }
    
    # Minimum content length for evaluation (characters)
    MIN_CONTENT_LENGTH = 100
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_cache = {}
    
    def evaluate_blog(self,
                     blog: Dict[str, Any],
                     theory_text: str,
                     theory_keywords: List[str],
                     concept_name: str) -> Tuple[Dict[str, Any], RelevanceMetrics]:
        """
        Comprehensively evaluate a blog for relevance.
        
        Args:
            blog: Blog dictionary with content
            theory_text: Theory content to compare against
            theory_keywords: Keywords from theory
            concept_name: Name of the concept
            
        Returns:
            Tuple of (evaluated_blog, metrics)
        """
        
        # Extract blog content
        blog_content = self._extract_content(blog)
        
        # Calculate individual metrics
        text_similarity = self._calculate_text_similarity(theory_text, blog_content)
        keyword_coverage = self._calculate_keyword_coverage(blog_content, theory_keywords)
        source_credibility = self._get_source_credibility(blog)
        content_depth = self._calculate_content_depth(blog_content)
        educational_value = self._calculate_educational_value(blog_content, theory_keywords)
        
        # Calculate overall score
        overall_score = (
            text_similarity * 0.25 +
            keyword_coverage * 0.25 +
            source_credibility * 0.20 +
            content_depth * 0.15 +
            educational_value * 0.15
        )
        
        # Create metrics object
        metrics = RelevanceMetrics(
            text_similarity=round(text_similarity, 3),
            keyword_coverage=round(keyword_coverage, 3),
            source_credibility=round(source_credibility, 3),
            content_depth=round(content_depth, 3),
            educational_value=round(educational_value, 3),
            overall_score=round(overall_score, 3)
        )
        
        # Add evaluation to blog
        evaluated_blog = blog.copy()
        evaluated_blog['evaluation'] = {
            'metrics': metrics.__dict__,
            'evaluated_at': datetime.now().isoformat(),
            'recommendation': self._get_recommendation(overall_score),
            'quality_level': self._get_quality_level(overall_score)
        }
        
        return evaluated_blog, metrics
    
    def _extract_content(self, blog: Dict[str, Any]) -> str:
        """Extract all text content from blog."""
        content_parts = [
            blog.get('title', ''),
            blog.get('content', ''),
            blog.get('snippet', ''),
            blog.get('summary', '')
        ]
        return ' '.join(str(part) for part in content_parts if part)
    
    def _calculate_text_similarity(self, theory_text: str, blog_content: str) -> float:
        """
        Calculate similarity between theory and blog content.
        Uses word overlap and semantic similarity.
        """
        if not theory_text or not blog_content:
            return 0.0
        
        # Normalize texts
        theory_words = set(theory_text.lower().split())
        blog_words = set(blog_content.lower().split())
        
        # Calculate Jaccard similarity
        if not theory_words or not blog_words:
            return 0.0
        
        intersection = len(theory_words & blog_words)
        union = len(theory_words | blog_words)
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Boost score if blog is longer (more detailed)
        length_factor = min(len(blog_content) / max(len(theory_text), 1), 1.0)
        
        return min(jaccard_similarity * 0.7 + length_factor * 0.3, 1.0)
    
    def _calculate_keyword_coverage(self, blog_content: str, keywords: List[str]) -> float:
        """
        Calculate how many theory keywords are covered in the blog.
        """
        if not keywords:
            return 0.5
        
        blog_lower = blog_content.lower()
        covered_keywords = sum(
            1 for keyword in keywords
            if keyword.lower() in blog_lower
        )
        
        coverage = covered_keywords / len(keywords)
        return min(coverage, 1.0)
    
    def _get_source_credibility(self, blog: Dict[str, Any]) -> float:
        """Get credibility score for the blog source."""
        source_key = blog.get('source_key', 'unknown')
        return self.SOURCE_CREDIBILITY.get(source_key, 0.50)
    
    def _calculate_content_depth(self, blog_content: str) -> float:
        """
        Calculate content depth based on length and structure.
        """
        # Check content length
        content_length = len(blog_content)
        length_score = min(content_length / 2000, 1.0)  # 2000 chars = full score
        
        # Check for structured content (sections, lists, etc.)
        structure_indicators = [
            blog_content.count('\n'),
            blog_content.count('##'),
            blog_content.count('- '),
            blog_content.count('* ')
        ]
        structure_score = min(sum(structure_indicators) / 10, 1.0)
        
        # Check for technical depth (code blocks, formulas, etc.)
        technical_indicators = [
            blog_content.count('```'),
            blog_content.count('```'),
            blog_content.count('$'),
            blog_content.count('equation')
        ]
        technical_score = min(sum(technical_indicators) / 5, 1.0)
        
        return (length_score * 0.5 + structure_score * 0.3 + technical_score * 0.2)
    
    def _calculate_educational_value(self, blog_content: str, keywords: List[str]) -> float:
        """
        Calculate educational value based on content quality indicators.
        """
        content_lower = blog_content.lower()
        
        # Check for educational indicators
        educational_phrases = [
            'learn', 'understand', 'explain', 'tutorial', 'guide',
            'example', 'practice', 'exercise', 'concept', 'theory',
            'definition', 'principle', 'method', 'approach', 'technique'
        ]
        
        phrase_count = sum(
            1 for phrase in educational_phrases
            if phrase in content_lower
        )
        
        phrase_score = min(phrase_count / 5, 1.0)
        
        # Check for clarity indicators
        clarity_indicators = [
            content_lower.count('?'),  # Questions
            content_lower.count('!'),  # Emphasis
            content_lower.count('note:'),
            content_lower.count('important:'),
            content_lower.count('tip:')
        ]
        
        clarity_score = min(sum(clarity_indicators) / 3, 1.0)
        
        # Check keyword density
        keyword_density = self._calculate_keyword_coverage(blog_content, keywords)
        
        return (phrase_score * 0.4 + clarity_score * 0.3 + keyword_density * 0.3)
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on score."""
        if score >= 0.85:
            return "Highly Recommended - Excellent match"
        elif score >= 0.70:
            return "Recommended - Good match"
        elif score >= 0.50:
            return "Acceptable - Moderate match"
        elif score >= 0.30:
            return "Consider - Weak match"
        else:
            return "Not Recommended - Poor match"
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.70:
            return "Good"
        elif score >= 0.50:
            return "Fair"
        elif score >= 0.30:
            return "Poor"
        else:
            return "Very Poor"
    
    def rank_blogs(self, blogs: List[Dict[str, Any]],
                  theory_text: str,
                  theory_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Rank multiple blogs by relevance.
        
        Args:
            blogs: List of blog dictionaries
            theory_text: Theory content
            theory_keywords: Theory keywords
            
        Returns:
            Sorted list of blogs with evaluation scores
        """
        evaluated_blogs = []
        
        for blog in blogs:
            evaluated_blog, metrics = self.evaluate_blog(
                blog,
                theory_text,
                theory_keywords,
                ""
            )
            evaluated_blogs.append(evaluated_blog)
        
        # Sort by overall score (descending)
        evaluated_blogs.sort(
            key=lambda x: x['evaluation']['metrics']['overall_score'],
            reverse=True
        )
        
        return evaluated_blogs
    
    def filter_by_threshold(self, blogs: List[Dict[str, Any]],
                           threshold: float = 0.60) -> List[Dict[str, Any]]:
        """
        Filter blogs by relevance threshold.
        
        Args:
            blogs: List of evaluated blogs
            threshold: Minimum relevance score (0-1)
            
        Returns:
            Filtered list of blogs above threshold
        """
        return [
            blog for blog in blogs
            if blog.get('evaluation', {}).get('metrics', {}).get('overall_score', 0) >= threshold
        ]

