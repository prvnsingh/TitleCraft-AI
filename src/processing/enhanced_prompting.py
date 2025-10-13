"""
Enhanced Prompt Engineering Module for TitleCraft AI

This module provides advanced, adaptive prompt generation that adjusts based on:
- Channel-specific patterns and performance data
- Content category and audience type
- Seasonal trends and recent performance
- Multi-dimensional pattern analysis
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..data.models import ChannelProfile, VideoData
from .llm_orchestrator import TitleGenerationRequest

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context information for adaptive prompting"""
    channel_type: str
    audience_level: str  # "beginner", "intermediate", "expert"
    content_category: str
    seasonal_factor: float
    urgency_level: str  # "low", "medium", "high"
    competitive_landscape: str
    performance_trend: str  # "improving", "stable", "declining"


@dataclass
class PromptTemplate:
    """Enhanced prompt template with metadata"""
    name: str
    template: str
    use_cases: List[str]
    channel_types: List[str]
    performance_weight: float
    requires_examples: bool
    min_data_quality: float


class EnhancedPromptEngineering:
    """
    Advanced prompt engineering system with adaptive templates.
    
    Features:
    - Context-aware prompt generation
    - Performance-optimized templates
    - Multi-dimensional pattern integration
    - Audience and content type adaptation
    - Seasonal and trending factors
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.context_analyzers = self._initialize_context_analyzers()
        
    def _get_base_template(self) -> str:
        """Reusable base template to reduce duplication"""
        return """You are a YouTube title optimization expert specializing in {content_type} content.

CHANNEL PROFILE: {{channel_profile}}
HIGH-PERFORMING EXAMPLES: {{similar_examples}}  
PATTERN ANALYSIS: {{pattern_insights}}

CONTENT CONTEXT:
- Topic: {{idea}}
- Category: {{content_category}}
- Trend: {{performance_trend}}
{specific_requirements}

FORMAT: Return exactly {{n_titles}} titles as JSON array:
["Title 1", "Title 2", "Title 3"]

Generate high-performing titles based on proven patterns:"""

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize enhanced prompt templates"""
        
        templates = {
            'educational_detailed': PromptTemplate(
                name="Educational Detailed",
                template=self._get_educational_template(),
                use_cases=["tutorials", "explanations", "how-to"],
                channel_types=["education", "technology", "science"],
                performance_weight=0.85,
                requires_examples=True,
                min_data_quality=0.7
            ),
            
            'entertainment_viral': PromptTemplate(
                name="Entertainment Viral",
                template=self._get_entertainment_template(),
                use_cases=["entertainment", "comedy", "reactions"],
                channel_types=["entertainment", "gaming", "lifestyle"],
                performance_weight=0.90,
                requires_examples=True,
                min_data_quality=0.6
            ),
            
            'news_urgent': PromptTemplate(
                name="News Urgent", 
                template=self._get_news_template(),
                use_cases=["breaking news", "updates", "analysis"],
                channel_types=["news", "politics", "current events"],
                performance_weight=0.80,
                requires_examples=False,
                min_data_quality=0.5
            ),
            
            'product_review': PromptTemplate(
                name="Product Review",
                template=self._get_review_template(),
                use_cases=["reviews", "comparisons", "buying guides"],
                channel_types=["tech", "consumer", "lifestyle"],
                performance_weight=0.85,
                requires_examples=True,
                min_data_quality=0.7
            ),
            
            'personal_story': PromptTemplate(
                name="Personal Story",
                template=self._get_story_template(),
                use_cases=["vlogs", "personal experience", "storytelling"],
                channel_types=["lifestyle", "personal", "travel"],
                performance_weight=0.75,
                requires_examples=True,
                min_data_quality=0.6
            ),
            
            'high_performance_optimization': PromptTemplate(
                name="High Performance Optimization",
                template=self._get_optimization_template(),
                use_cases=["viral content", "growth hacking", "engagement"],
                channel_types=["any"],
                performance_weight=0.95,
                requires_examples=True,
                min_data_quality=0.8
            )
        }
        
        return templates
    
    def _get_educational_template(self) -> str:
        """Educational content template focused on learning outcomes"""
        base_template = self._get_base_template()
        specific_requirements = """
EDUCATIONAL REQUIREMENTS:
- Use learning patterns: "How to", "Complete Guide", "Step-by-step"
- Include skill indicators: "Beginner", "Advanced", "Ultimate" 
- Promise measurable outcomes and clear value
- Address learning pain points with urgency

SUCCESS FACTORS: Action-oriented, outcome-focused, searchable terms
"""
        return base_template.format(
            content_type="EDUCATIONAL",
            specific_requirements=specific_requirements
        )

    def _get_entertainment_template(self) -> str:
        """Entertainment template optimized for viral potential"""
        base_template = self._get_base_template()
        specific_requirements = """
ENTERTAINMENT REQUIREMENTS:
- Use high-emotion triggers and viral patterns
- Create curiosity gaps with payoff promises
- Include social proof and trending elements
- Leverage FOMO, contrast, and pattern interrupts

SUCCESS FACTORS: Maximum CTR, shareability, emotional triggers
"""
        return base_template.format(
            content_type="ENTERTAINMENT", 
            specific_requirements=specific_requirements
        )

    def _get_news_template(self) -> str:
        """News template for current events and breaking coverage"""
        base_template = self._get_base_template()
        specific_requirements = """
NEWS REQUIREMENTS:
- Lead with strongest angle using urgency indicators
- Use active voice, present tense, strong action verbs
- Include credible sources and key stakeholders
- Address core questions: who, what, when, where
- Balance immediacy with verified information

SUCCESS FACTORS: Newsworthy elements, verified sources, legitimate engagement
"""
        return base_template.format(
            content_type="NEWS",
            specific_requirements=specific_requirements
        )

    def _get_review_template(self) -> str:
        """Review template for product evaluations and buying guides"""
        base_template = self._get_base_template()
        specific_requirements = """
REVIEW REQUIREMENTS:
- Include clear indicators: "Review", "Test", "Comparison"
- Address key buying concerns and pain points
- Use trust-building language with buyer-intent keywords
- Promise thorough evaluation and recommendations
- Balance objectivity with engagement

SUCCESS FACTORS: Purchase influence, comparative elements, trust-building
"""
        return base_template.format(
            content_type="REVIEW",
            specific_requirements=specific_requirements
        )

    def _get_story_template(self) -> str:
        """Storytelling template for personal narratives and experiences"""
        base_template = self._get_base_template()
        specific_requirements = """
STORYTELLING REQUIREMENTS:
- Lead with most compelling story elements
- Use authentic, conversational language
- Include emotional hooks and universal themes  
- Promise relatable experiences and takeaways
- Create curiosity about personal outcomes

SUCCESS FACTORS: Emotional connection, relatability, transformation elements
"""
        return base_template.format(
            content_type="STORYTELLING",
            specific_requirements=specific_requirements
        )

    def _get_optimization_template(self) -> str:
        """Advanced optimization template for maximum performance"""
        base_template = self._get_base_template() 
        specific_requirements = """
OPTIMIZATION REQUIREMENTS:
- Engineer for MAXIMUM performance using proven patterns
- Apply psychological triggers specific to this audience
- Use advanced pattern combinations for compound effectiveness
- Target optimal length and trending alignment
- Include performance predictions and confidence scores

SUCCESS FACTORS: Elite patterns, psychological optimization, competitive edge

FORMAT: JSON with performance metrics:
[{{"title": "...", "predicted_ctr": 8.5, "confidence": 0.92}}]
"""
        return base_template.format(
            content_type="OPTIMIZATION",
            specific_requirements=specific_requirements
        )

    def _initialize_context_analyzers(self) -> Dict[str, Any]:
        """Initialize context analysis functions"""
        return {
            'channel_type': self._analyze_channel_type,
            'audience_level': self._analyze_audience_level, 
            'content_category': self._analyze_content_category,
            'seasonal_factors': self._analyze_seasonal_factors,
            'competitive_landscape': self._analyze_competitive_landscape,
            'performance_trend': self._analyze_performance_trend
        }
    
    def create_adaptive_prompt(self,
                              request: TitleGenerationRequest,
                              channel_profile: ChannelProfile,
                              similar_examples: List[VideoData],
                              enhanced_profile: Dict[str, Any] = None) -> str:
        """
        Create highly adaptive prompt based on comprehensive analysis
        
        Args:
            request: Title generation request
            channel_profile: Channel profile data
            similar_examples: Similar high-performing videos
            enhanced_profile: Enhanced pattern analysis data
            
        Returns:
            Optimized prompt string
        """
        
        # Analyze context
        context = self._build_prompt_context(request, channel_profile, enhanced_profile)
        
        # Select optimal template
        template = self._select_optimal_template(context, channel_profile, enhanced_profile)
        
        # Build comprehensive prompt
        prompt = self._build_enhanced_prompt(
            template, request, channel_profile, similar_examples, enhanced_profile, context
        )
        
        return prompt
    
    def _build_prompt_context(self,
                             request: TitleGenerationRequest,
                             channel_profile: ChannelProfile,
                             enhanced_profile: Dict[str, Any] = None) -> PromptContext:
        """Build comprehensive context for prompt generation"""
        
        # Analyze various context factors
        channel_type = self._analyze_channel_type(channel_profile, enhanced_profile)
        audience_level = self._analyze_audience_level(channel_profile, enhanced_profile)
        content_category = self._analyze_content_category(request.idea, enhanced_profile)
        seasonal_factor = self._analyze_seasonal_factors(datetime.now())
        urgency_level = self._analyze_urgency_level(request, enhanced_profile)
        competitive_landscape = self._analyze_competitive_landscape(channel_profile)
        performance_trend = self._analyze_performance_trend(channel_profile)
        
        return PromptContext(
            channel_type=channel_type,
            audience_level=audience_level,
            content_category=content_category,
            seasonal_factor=seasonal_factor,
            urgency_level=urgency_level,
            competitive_landscape=competitive_landscape,
            performance_trend=performance_trend
        )
    
    def _select_optimal_template(self,
                                context: PromptContext,
                                channel_profile: ChannelProfile,
                                enhanced_profile: Dict[str, Any] = None) -> PromptTemplate:
        """Select the most appropriate template based on context"""
        
        # Calculate template scores
        template_scores = {}
        
        for name, template in self.templates.items():
            score = 0.0
            
            # Channel type match
            if context.channel_type in template.channel_types or "any" in template.channel_types:
                score += 0.3
            
            # Content category match
            if context.content_category in template.use_cases:
                score += 0.2
            
            # Performance weight
            score += template.performance_weight * 0.3
            
            # Data quality check
            data_quality = self._calculate_data_quality(channel_profile, enhanced_profile)
            if data_quality >= template.min_data_quality:
                score += 0.2
            
            template_scores[name] = score
        
        # Select highest scoring template
        best_template_name = max(template_scores, key=template_scores.get)
        return self.templates[best_template_name]
    
    def _build_enhanced_prompt(self,
                              template: PromptTemplate,
                              request: TitleGenerationRequest,
                              channel_profile: ChannelProfile,
                              similar_examples: List[VideoData],
                              enhanced_profile: Dict[str, Any],
                              context: PromptContext) -> str:
        """Build the final enhanced prompt"""
        
        # Prepare all template variables
        template_vars = {
            'idea': request.idea,
            'n_titles': request.n_titles,
            'channel_profile': self._format_enhanced_channel_profile(channel_profile, enhanced_profile),
            'similar_examples': self._format_contextual_examples(similar_examples, context),
            'pattern_insights': self._generate_pattern_insights(enhanced_profile, context),
            'target_length': self._calculate_optimal_length(channel_profile, enhanced_profile),
            'audience_level': context.audience_level,
            'content_category': context.content_category,
            'seasonal_factor': context.seasonal_factor,
            'performance_trend': context.performance_trend,
            'successful_patterns': self._extract_successful_patterns(enhanced_profile),
            'emotional_triggers': self._identify_emotional_triggers(enhanced_profile),
            'viral_patterns': self._extract_viral_patterns(enhanced_profile),
            'trending_elements': self._identify_trending_elements(context),
            'performance_target': self._calculate_performance_target(channel_profile)
        }
        
        # Fill template with error handling
        try:
            filled_template = template.template.format(**template_vars)
            return filled_template
        except KeyError as e:
            logger.warning(f"Template variable missing: {e}, using default template")
            return self._build_fallback_prompt(request, channel_profile, similar_examples)
    
    # Context analyzer methods
    def _analyze_channel_type(self, profile: ChannelProfile, enhanced: Dict = None) -> str:
        """Analyze and determine channel type"""
        if hasattr(profile, 'channel_type'):
            return profile.channel_type
        
        # Infer from content themes if available
        if enhanced and 'content_themes' in enhanced:
            themes = enhanced['content_themes']
            # Logic to determine type from themes
            return "general"  # Default
        
        return "general"
    
    def _analyze_audience_level(self, profile: ChannelProfile, enhanced: Dict = None) -> str:
        """Analyze target audience sophistication level"""
        # Analyze title complexity, technical terms, etc.
        avg_length = profile.title_patterns.avg_length_words
        
        if avg_length > 12:
            return "expert"
        elif avg_length > 8:
            return "intermediate"
        else:
            return "beginner"
    
    def _analyze_content_category(self, idea: str, enhanced: Dict = None) -> str:
        """Analyze content category from idea"""
        idea_lower = idea.lower()
        
        # Educational keywords
        if any(word in idea_lower for word in ['how', 'learn', 'tutorial', 'guide', 'explain']):
            return "educational"
        
        # Entertainment keywords  
        if any(word in idea_lower for word in ['funny', 'reaction', 'prank', 'comedy', 'entertainment']):
            return "entertainment"
        
        # Review keywords
        if any(word in idea_lower for word in ['review', 'test', 'vs', 'comparison', 'versus']):
            return "review"
        
        return "general"
    
    def _analyze_seasonal_factors(self, current_date: datetime) -> float:
        """Analyze seasonal relevance factors"""
        month = current_date.month
        
        # Holiday seasons get boost
        if month in [11, 12]:  # November, December
            return 1.2
        elif month in [1, 9]:  # January (resolutions), September (back to school)
            return 1.1
        else:
            return 1.0
    
    def _analyze_urgency_level(self, request: TitleGenerationRequest, enhanced: Dict = None) -> str:
        """Analyze content urgency level"""
        idea_lower = request.idea.lower()
        
        if any(word in idea_lower for word in ['breaking', 'urgent', 'now', 'today', 'just happened']):
            return "high"
        elif any(word in idea_lower for word in ['update', 'news', 'latest', 'recent']):
            return "medium"
        else:
            return "low"
    
    def _analyze_competitive_landscape(self, profile: ChannelProfile) -> str:
        """Analyze competitive landscape intensity"""
        # Based on performance thresholds and view distribution
        cv = profile.stats.std_views / profile.stats.avg_views if profile.stats.avg_views > 0 else 0
        
        if cv > 2.0:
            return "highly_competitive"
        elif cv > 1.0:
            return "moderately_competitive"
        else:
            return "low_competition"
    
    def _analyze_performance_trend(self, profile: ChannelProfile) -> str:
        """Analyze recent performance trends"""
        # This would require time-series data
        # For now, return stable as default
        return "stable"
    
    # Helper methods for template building
    def _format_enhanced_channel_profile(self, profile: ChannelProfile, enhanced: Dict) -> str:
        """Format enhanced channel profile for prompt"""
        basic_info = f"""- Channel Type: {profile.channel_type}
- Total Videos: {profile.stats.total_videos}
- Average Views: {profile.stats.avg_views:,.0f}
- High Performer Threshold: {profile.stats.high_performer_threshold:,.0f}"""
        
        if enhanced:
            pattern_info = f"""
- Question Format: {profile.title_patterns.question_ratio:.0%}
- Numeric Elements: {profile.title_patterns.numeric_ratio:.0%}
- Superlative Usage: {profile.title_patterns.superlative_ratio:.0%}
- Emotional Hooks: {profile.title_patterns.emotional_hook_ratio:.0%}"""
            
            return basic_info + pattern_info
        
        return basic_info
    
    def _format_contextual_examples(self, examples: List[VideoData], context: PromptContext) -> str:
        """Format examples with contextual relevance"""
        if not examples:
            return "No similar examples available."
        
        formatted = []
        for i, example in enumerate(examples[:5], 1):
            formatted.append(f'{i}. "{example.title}" - {example.views_in_period:,} views')
        
        return "\\n".join(formatted)
    
    def _generate_pattern_insights(self, enhanced: Dict, context: PromptContext) -> str:
        """Generate pattern insights from enhanced analysis"""
        if not enhanced:
            return "Limited pattern data available."
        
        insights = []
        
        # Structural patterns
        if 'structural_patterns' in enhanced:
            top_patterns = sorted(
                enhanced['structural_patterns'].items(),
                key=lambda x: x[1].get('performance_vs_average', 0),
                reverse=True
            )[:3]
            
            for pattern, data in top_patterns:
                insights.append(f"• {pattern}: {data['usage_rate']:.0%} usage, {data['performance_vs_average']:.1f}x average performance")
        
        return "\\n".join(insights) if insights else "Pattern analysis in progress."
    
    def _calculate_optimal_length(self, profile: ChannelProfile, enhanced: Dict) -> int:
        """Calculate optimal title length for this channel"""
        return int(profile.title_patterns.avg_length_words)
    
    def _extract_successful_patterns(self, enhanced: Dict) -> str:
        """Extract most successful patterns"""
        if not enhanced or 'pattern_correlations' not in enhanced:
            return "Pattern analysis in progress."
        
        correlations = enhanced['pattern_correlations']
        top_patterns = sorted(correlations, key=lambda x: x.get('correlation_score', 0), reverse=True)[:3]
        
        return ", ".join([f"{p['pattern_name']}" for p in top_patterns])
    
    def _identify_emotional_triggers(self, enhanced: Dict) -> str:
        """Identify top emotional triggers"""
        if not enhanced or 'emotional_triggers' not in enhanced:
            return "curiosity, excitement, urgency"
        
        triggers = enhanced['emotional_triggers']
        top_triggers = sorted(
            triggers.items(),
            key=lambda x: x[1].get('performance_vs_average', 0),
            reverse=True
        )[:3]
        
        return ", ".join([trigger for trigger, _ in top_triggers])
    
    def _extract_viral_patterns(self, enhanced: Dict) -> str:
        """Extract patterns associated with viral content"""
        # This would analyze top-performing content patterns
        return "shock value, curiosity gaps, social proof"
    
    def _identify_trending_elements(self, context: PromptContext) -> str:
        """Identify current trending elements"""
        # This would integrate with trending APIs
        return "current events, seasonal topics, platform features"
    
    def _calculate_performance_target(self, profile: ChannelProfile) -> str:
        """Calculate realistic performance target"""
        return f"{profile.stats.high_performer_threshold:,.0f}"
    
    def _calculate_data_quality(self, profile: ChannelProfile, enhanced: Dict) -> float:
        """Calculate data quality score"""
        base_score = min(1.0, profile.stats.total_videos / 50.0)
        
        if enhanced and enhanced.get('profile_metadata', {}).get('data_quality_score'):
            return enhanced['profile_metadata']['data_quality_score']
        
        return base_score
    
    def _build_fallback_prompt(self,
                              request: TitleGenerationRequest,
                              profile: ChannelProfile,
                              examples: List[VideoData]) -> str:
        """Build a simple fallback prompt if template fails"""
        return f"""Create {request.n_titles} YouTube titles for: {request.idea}

Channel averages {profile.stats.avg_views:,.0f} views and uses {profile.title_patterns.avg_length_words:.0f} word titles.

Return as JSON array: ["Title 1", "Title 2", ...]"""