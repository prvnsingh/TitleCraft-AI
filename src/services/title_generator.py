from openai import OpenAI

# Title Generator Class
from src.data_module.data_processor import DataLoader
from src.data_module.data_processor import ChannelAnalysis, GeneratedTitle
from typing import List, Dict, Any, Optional
import os


class TitleGenerator:
    """Simple title generator using pattern analysis and LLM"""
    
    def __init__(self, api_key: str = None):
        self.data_loader = DataLoader()
        
        # Initialize OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        self.client = OpenAI(api_key=api_key)
    
    def generate_titles(self, channel_id: str, idea: str, n_titles: int = 4) -> List[GeneratedTitle]:
        """Generate titles for a video idea based on channel patterns"""
        try:
            # Analyze channel patterns
            channel_analysis = self.data_loader.analyze_channel(channel_id)
            
            # Create prompt based on patterns
            prompt = self._create_prompt(channel_analysis, idea, n_titles)
            
            # Generate titles using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a YouTube title optimization expert. Create engaging titles based on successful patterns from the channel's history."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Parse response
            generated_titles = self._parse_response(response.choices[0].message.content, channel_analysis)
            
            return generated_titles[:n_titles]
            
        except Exception:
            # Fallback to pattern-based titles if LLM fails
            return self._generate_fallback_titles(idea, n_titles)
    
    def _create_prompt(self, analysis: ChannelAnalysis, idea: str, n_titles: int) -> str:
        """Create prompt for title generation"""
        
        top_titles = [video.title for video in analysis.top_performers[:5]]
        patterns = analysis.patterns
        
        prompt = f"""
Based on this YouTube channel's most successful titles and patterns, generate {n_titles} engaging titles for a new video.

CHANNEL PERFORMANCE DATA:
- Total videos: {analysis.total_videos}
- Average views: {analysis.avg_views:,.0f}

TOP PERFORMING TITLES:
{chr(10).join(f"• {title} ({video.views_in_period:,} views)" for video, title in zip(analysis.top_performers[:5], top_titles))}

SUCCESSFUL PATTERNS IDENTIFIED:
- Average word count: {patterns.get('avg_length', 0):.1f} words
- Question titles: {patterns.get('question_titles', 0):.0%} of top performers
- Titles with numbers: {patterns.get('numeric_titles', 0):.0%} of top performers  
- Exclamation usage: {patterns.get('exclamation_titles', 0):.0%} of top performers
- Common successful words: {', '.join([word for word, count in patterns.get('common_words', [])])}

NEW VIDEO IDEA: "{idea}"

Please generate {n_titles} titles that follow the successful patterns from this channel. For each title, provide:
1. The title text
2. Brief reasoning explaining which patterns it uses and why it should perform well

Format as:
TITLE 1: [title text]
REASONING: [explanation of patterns used and expected performance]

TITLE 2: [title text] 
REASONING: [explanation of patterns used and expected performance]

[Continue for all {n_titles} titles]
"""
        return prompt
    
    def _parse_response(self, response_text: str, analysis: ChannelAnalysis) -> List[GeneratedTitle]:
        """Parse LLM response into GeneratedTitle objects"""
        titles = []
        lines = response_text.strip().split('\n')
        
        current_title = None
        current_reasoning = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('TITLE'):
                if current_title and current_reasoning:
                    # Save previous title
                    titles.append(GeneratedTitle(
                        title=current_title,
                        reasoning=current_reasoning,
                        confidence=self._calculate_confidence(current_title, analysis)
                    ))
                
                # Extract new title
                current_title = line.split(':', 1)[1].strip()
                current_reasoning = None
                
            elif line.startswith('REASONING'):
                current_reasoning = line.split(':', 1)[1].strip()
        
        # Don't forget the last title
        if current_title and current_reasoning:
            titles.append(GeneratedTitle(
                title=current_title,
                reasoning=current_reasoning,
                confidence=self._calculate_confidence(current_title, analysis)
            ))
        
        return titles
    
    def _calculate_confidence(self, title: str, analysis: ChannelAnalysis) -> float:
        """Calculate confidence score based on pattern matching"""
        patterns = analysis.patterns
        score = 0.5  # Base score
        
        # Word count alignment
        word_count = len(title.split())
        target_length = patterns.get('avg_length', 8)
        if abs(word_count - target_length) <= 2:
            score += 0.2
        
        # Pattern matching bonuses
        if '?' in title and patterns.get('question_titles', 0) > 0.2:
            score += 0.15
        
        if any(char.isdigit() for char in title) and patterns.get('numeric_titles', 0) > 0.3:
            score += 0.15
        
        # Common words bonus
        common_words = [word for word, count in patterns.get('common_words', [])]
        title_words = title.lower().split()
        if any(word in ' '.join(title_words) for word in common_words[:3]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_fallback_titles(self, idea: str, n_titles: int) -> List[GeneratedTitle]:
        """Generate fallback titles if LLM fails"""
        # Simple pattern-based title generation
        templates = [
            f"How to {idea}",
            f"Why {idea} Works",
            f"The Truth About {idea}",
            f"{idea}: What You Need to Know"
        ]
        
        fallback_titles = []
        for i, template in enumerate(templates[:n_titles]):
            fallback_titles.append(GeneratedTitle(
                title=template,
                reasoning="Pattern-based title following channel's successful format (fallback generation due to LLM unavailability)",
                confidence=0.3
            ))
        
        return fallback_titles
    
    def generate_titles_fallback(self, channel_id: str, idea: str, n_titles: int = 4) -> List[GeneratedTitle]:
        """Generate titles using fallback method only (no OpenAI required)"""
        try:
            # Analyze channel patterns
            channel_analysis = self.data_loader.analyze_channel(channel_id)
            
            # Generate pattern-based titles
            return self._generate_pattern_based_titles(idea, channel_analysis, n_titles)
            
        except ValueError:
            # If channel not found, use generic fallback
            return self._generate_fallback_titles(idea, n_titles)
    
    def _generate_pattern_based_titles(self, idea: str, analysis: ChannelAnalysis, n_titles: int) -> List[GeneratedTitle]:
        """Generate titles based on channel patterns without LLM"""
        patterns = analysis.patterns
        top_titles = [video.title for video in analysis.top_performers[:3]]
        
        # Create pattern-based templates
        templates = []
        
        # Use successful question format if channel uses questions
        if patterns.get('question_titles', 0) > 0.2:
            templates.extend([
                f"How Does {idea}?",
                f"Why {idea}?", 
                f"What Makes {idea} Work?"
            ])
        
        # Use numbers if channel uses numeric titles
        if patterns.get('numeric_titles', 0) > 0.3:
            templates.extend([
                f"5 Facts About {idea}",
                f"The 3 Secrets of {idea}",
                f"10 Things You Need to Know About {idea}"
            ])
        
        # Use exclamations if channel uses them
        if patterns.get('exclamation_titles', 0) > 0.2:
            templates.extend([
                f"{idea}!",
                f"Amazing {idea}!",
                f"Incredible {idea} You Won't Believe!"
            ])
        
        # Always include some safe patterns
        templates.extend([
            f"The Complete Guide to {idea}",
            f"Understanding {idea}",
            f"The Truth About {idea}",
            f"{idea}: What You Need to Know"
        ])
        
        # Generate titles
        generated_titles = []
        for i, template in enumerate(templates[:n_titles]):
            confidence = 0.5 + (0.2 if any(word in template.lower() for word, count in patterns.get('common_words', [])) else 0)
            generated_titles.append(GeneratedTitle(
                title=template,
                reasoning=f"Pattern-based title using channel's successful format - {patterns.get('avg_length', 6):.1f} avg words, follows top performers like '{top_titles[0] if top_titles else 'N/A'}'",
                confidence=min(confidence, 1.0)
            ))
        
        return generated_titles
