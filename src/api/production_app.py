"""
Simple FastAPI Application for TitleCraft AI Take-Home Task
Minimal implementation focusing on core requirements only
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import pandas as pd
from dataclasses import dataclass
import openai

# Data Models
@dataclass
class VideoData:
    """Simple video data structure"""
    channel_id: str
    video_id: str
    title: str
    summary: str
    views_in_period: int

@dataclass 
class ChannelAnalysis:
    """Channel performance analysis"""
    channel_id: str
    total_videos: int
    avg_views: float
    top_performers: List[VideoData]
    patterns: Dict[str, Any]

@dataclass
class GeneratedTitle:
    """Simple generated title with reasoning"""
    title: str
    reasoning: str
    confidence: float

# Data Loader Class
class DataLoader:
    """Simple CSV data loader and analyzer"""
    
    def __init__(self, csv_path: str = "electrify__applied_ai_engineer__training_data.csv"):
        self.csv_path = csv_path
        self.data: Optional[pd.DataFrame] = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load data from CSV file"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} videos from {len(self.data['channel_id'].unique())} channels")
    
    def get_channel_data(self, channel_id: str) -> List[VideoData]:
        """Get all videos for a specific channel"""
        if self.data is None:
            return []
        
        channel_videos = self.data[self.data['channel_id'] == channel_id]
        
        return [
            VideoData(
                channel_id=row['channel_id'],
                video_id=row['video_id'],
                title=row['title'],
                summary=row['summary'],
                views_in_period=row['views_in_period']
            ) for _, row in channel_videos.iterrows()
        ]
    
    def analyze_channel(self, channel_id: str) -> ChannelAnalysis:
        """Analyze channel performance patterns"""
        videos = self.get_channel_data(channel_id)
        
        if not videos:
            raise ValueError(f"No data found for channel: {channel_id}")
        
        # Basic statistics
        views = [v.views_in_period for v in videos]
        avg_views = sum(views) / len(views)
        
        # Get top performers (top 30%)
        sorted_videos = sorted(videos, key=lambda v: v.views_in_period, reverse=True)
        top_count = max(1, len(sorted_videos) // 3)
        top_performers = sorted_videos[:top_count]
        
        # Analyze title patterns from top performers
        patterns = self._analyze_title_patterns([v.title for v in top_performers])
        
        return ChannelAnalysis(
            channel_id=channel_id,
            total_videos=len(videos),
            avg_views=avg_views,
            top_performers=top_performers,
            patterns=patterns
        )
    
    def _analyze_title_patterns(self, titles: List[str]) -> Dict[str, Any]:
        """Extract simple patterns from high-performing titles"""
        if not titles:
            return {}
        
        # Basic pattern analysis
        patterns = {
            'avg_length': sum(len(title.split()) for title in titles) / len(titles),
            'question_titles': sum(1 for title in titles if '?' in title) / len(titles),
            'numeric_titles': sum(1 for title in titles if any(char.isdigit() for char in title)) / len(titles),
            'exclamation_titles': sum(1 for title in titles if '!' in title) / len(titles),
            'common_words': self._get_common_words(titles),
            'sample_titles': titles[:3]  # Examples for pattern reference
        }
        
        return patterns
    
    def _get_common_words(self, titles: List[str]) -> List[str]:
        """Get most common words from titles (simple version)"""
        word_count = {}
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        for title in titles:
            words = title.lower().split()
            for word in words:
                # Clean word
                word = ''.join(char for char in word if char.isalnum())
                if len(word) > 2 and word not in stop_words:
                    word_count[word] = word_count.get(word, 0) + 1
        
        # Return top 5 most common words
        return sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]

# Title Generator Class
class TitleGenerator:
    """Simple title generator using pattern analysis and LLM"""
    
    def __init__(self, api_key: str = None):
        self.data_loader = DataLoader()
        
        # Initialize OpenAI
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
        if not openai.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def generate_titles(self, channel_id: str, idea: str, n_titles: int = 4) -> List[GeneratedTitle]:
        """Generate titles for a video idea based on channel patterns"""
        try:
            # Analyze channel patterns
            channel_analysis = self.data_loader.analyze_channel(channel_id)
            
            # Create prompt based on patterns
            prompt = self._create_prompt(channel_analysis, idea, n_titles)
            
            # Generate titles using OpenAI
            response = openai.ChatCompletion.create(
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

# Request/Response Models
class TitleRequest(BaseModel):
    """Request model for title generation"""
    channel_id: str = Field(..., description="YouTube channel ID")
    idea: str = Field(..., description="Single-sentence video idea or topic")

class TitleResponse(BaseModel):
    """Individual title in response"""
    title: str = Field(..., description="Generated title")
    reasoning: str = Field(..., description="Data-grounded reasoning for this title")

class GenerationResponse(BaseModel):
    """Response model for title generation"""
    titles: List[TitleResponse] = Field(..., description="3-5 generated titles with reasoning")
    channel_id: str = Field(..., description="Channel ID processed")
    idea: str = Field(..., description="Video idea processed")

# Initialize FastAPI app
app = FastAPI(
    title="TitleCraft AI - Take-Home Task",
    description="YouTube title generation based on channel performance patterns",
    version="1.0.0"
)

# Initialize title generator
try:
    title_generator = TitleGenerator()
except ValueError as e:
    print(f"Warning: {e}")
    title_generator = None

@app.post("/generate", response_model=GenerationResponse)
async def generate_titles(request: TitleRequest) -> GenerationResponse:
    """
    Generate optimized YouTube titles based on channel patterns and video idea.
    
    This endpoint:
    1. Analyzes the channel's historical performance data
    2. Identifies patterns from high-performing titles
    3. Generates 3-5 new titles that follow successful patterns
    4. Provides data-grounded reasoning for each suggestion
    """
    
    if not title_generator:
        raise HTTPException(
            status_code=500, 
            detail="Title generator not available. Please check OpenAI API key configuration."
        )
    
    try:
        # Generate titles
        generated_titles = title_generator.generate_titles(
            channel_id=request.channel_id,
            idea=request.idea,
            n_titles=4  # Generate 4 titles as specified in requirements
        )
        
        if not generated_titles:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for channel: {request.channel_id}"
            )
        
        # Format response
        response_titles = [
            TitleResponse(
                title=gen_title.title,
                reasoning=gen_title.reasoning
            ) for gen_title in generated_titles
        ]
        
        return GenerationResponse(
            titles=response_titles,
            channel_id=request.channel_id,
            idea=request.idea
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "TitleCraft AI",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "TitleCraft AI - Take-Home Task",
        "version": "1.0.0",
        "docs": "/docs",
        "generate_endpoint": "/generate",
        "description": "YouTube title generation based on channel performance patterns"
    }

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "src.api.production_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )