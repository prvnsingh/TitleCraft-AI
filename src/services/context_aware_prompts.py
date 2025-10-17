"""
Context-Aware Prompt Selector
Selects optimal prompts and parameters based on channel analysis
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .pattern_discovery import IntelligentPatterns
from .logger_config import (
    titlecraft_logger, 
    log_execution_flow, 
    log_data_analysis, 
    log_context_aware_decision
)


class PromptStrategy(Enum):
    """Different prompt strategies based on channel characteristics"""
    HIGH_VOLUME_EDUCATIONAL = "high_volume_educational"
    HIGH_VOLUME_ENTERTAINMENT = "high_volume_entertainment"
    MEDIUM_VOLUME_MIXED = "medium_volume_mixed"
    LOW_VOLUME_ADAPTIVE = "low_volume_adaptive"
    DEFAULT = "default"


@dataclass
class AdaptiveParameters:
    """LLM parameters adapted for specific content types"""
    temperature: float
    max_tokens: int
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None


@dataclass
class ContextualPrompt:
    """Contextual prompt with adaptive parameters"""
    system_prompt: str
    user_prompt_template: str
    parameters: AdaptiveParameters
    strategy: PromptStrategy


class ContextAwarePromptSelector:
    """
    Intelligently selects prompts and parameters based on channel analysis
    """
    
    def __init__(self):
        self.logger = titlecraft_logger.get_logger("context_aware_prompts")
        self.prompt_strategies = self._initialize_strategies()
        
        self.logger.info("ContextAwarePromptSelector initialized", extra={
            'extra_fields': {
                'component': 'context_aware_prompts',
                'action': 'initialization',
                'strategies_count': len(self.prompt_strategies)
            }
        })
    
    @log_execution_flow("context_selection", "context_aware_prompts")
    def select_optimal_context(self, patterns: IntelligentPatterns) -> ContextualPrompt:
        """
        Select the optimal prompt strategy and parameters based on patterns
        """
        self.logger.info("Starting optimal context selection", extra={
            'extra_fields': {
                'component': 'context_aware_prompts',
                'action': 'context_selection_start',
                'channel_type': patterns.channel_type,
                'content_style': patterns.content_style,
                'confidence_score': patterns.confidence_score,
                'avg_word_count': patterns.avg_word_count
            }
        })
        
        strategy = self._determine_strategy(patterns)
        
        log_context_aware_decision(
            decision_type="prompt_strategy_selection",
            reasoning=f"Selected {strategy.value} based on channel_type={patterns.channel_type}, content_style={patterns.content_style}",
            confidence=patterns.confidence_score,
            component="context_aware_prompts"
        )
        
        base_prompt = self._get_strategy_prompt(strategy)
        
        # Adapt parameters based on patterns
        parameters = self._adapt_parameters(patterns, strategy)
        
        self.logger.info("Parameters adapted for context", extra={
            'extra_fields': {
                'component': 'context_aware_prompts',
                'action': 'parameter_adaptation',
                'strategy': strategy.value,
                'temperature': parameters.temperature,
                'max_tokens': parameters.max_tokens,
                'top_p': parameters.top_p,
                'presence_penalty': parameters.presence_penalty
            }
        })
        
        # Customize prompt content based on patterns
        system_prompt = self._customize_system_prompt(base_prompt.system_prompt, patterns)
        user_prompt = self._customize_user_prompt(base_prompt.user_prompt_template, patterns)
        
        self.logger.info("Prompts customized based on patterns", extra={
            'extra_fields': {
                'component': 'context_aware_prompts',
                'action': 'prompt_customization',
                'system_prompt_length': len(system_prompt),
                'user_prompt_length': len(user_prompt),
                'top_keywords': patterns.top_keywords[:3] if hasattr(patterns, 'top_keywords') else [],
                'pattern_weights': {
                    'word_count': getattr(patterns.pattern_weights, 'word_count_weight', None) if hasattr(patterns, 'pattern_weights') else None,
                    'question': getattr(patterns.pattern_weights, 'question_weight', None) if hasattr(patterns, 'pattern_weights') else None,
                    'numeric': getattr(patterns.pattern_weights, 'numeric_weight', None) if hasattr(patterns, 'pattern_weights') else None
                }
            }
        })
        
        contextual_prompt = ContextualPrompt(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            parameters=parameters,
            strategy=strategy
        )
        
        self.logger.info("Context selection completed", extra={
            'extra_fields': {
                'component': 'context_aware_prompts',
                'action': 'context_selection_complete',
                'final_strategy': strategy.value,
                'final_temperature': parameters.temperature,
                'final_max_tokens': parameters.max_tokens
            }
        })
        
        return contextual_prompt
    
    def _determine_strategy(self, patterns: IntelligentPatterns) -> PromptStrategy:
        """Determine the best strategy based on channel patterns"""
        
        if patterns.channel_type == "high_volume":
            if patterns.content_style == "educational":
                return PromptStrategy.HIGH_VOLUME_EDUCATIONAL
            elif patterns.content_style == "entertainment":
                return PromptStrategy.HIGH_VOLUME_ENTERTAINMENT
            else:
                return PromptStrategy.MEDIUM_VOLUME_MIXED
                
        elif patterns.channel_type == "medium_volume":
            return PromptStrategy.MEDIUM_VOLUME_MIXED
            
        elif patterns.channel_type == "low_volume":
            return PromptStrategy.LOW_VOLUME_ADAPTIVE
            
        else:
            return PromptStrategy.DEFAULT
    
    def _get_strategy_prompt(self, strategy: PromptStrategy) -> ContextualPrompt:
        """Get base prompt for strategy"""
        return self.prompt_strategies[strategy]
    
    def _adapt_parameters(self, patterns: IntelligentPatterns, 
                         strategy: PromptStrategy) -> AdaptiveParameters:
        """Adapt LLM parameters based on patterns and strategy"""
        
        # Base parameters from strategy
        base_params = self.prompt_strategies[strategy].parameters
        
        # Adjust temperature based on confidence
        # Higher confidence -> lower temperature (more focused)
        # Lower confidence -> higher temperature (more creative)
        temperature_adjustment = (1.0 - patterns.confidence_score) * 0.3
        adjusted_temperature = min(1.0, max(0.1, base_params.temperature + temperature_adjustment))
        
        # Adjust max_tokens based on channel characteristics
        token_multiplier = 1.0
        if patterns.channel_type == "high_volume":
            token_multiplier = 1.2  # More detailed analysis possible
        elif patterns.channel_type == "low_volume":
            token_multiplier = 0.9  # More concise due to limited data
        
        adjusted_max_tokens = int(base_params.max_tokens * token_multiplier)
        
        return AdaptiveParameters(
            temperature=adjusted_temperature,
            max_tokens=adjusted_max_tokens,
            top_p=base_params.top_p,
            presence_penalty=base_params.presence_penalty
        )
    
    def _customize_system_prompt(self, base_prompt: str, patterns: IntelligentPatterns) -> str:
        """Customize system prompt based on discovered patterns"""
        
        # Add channel-specific context
        channel_context = f"""
        
CHANNEL CHARACTERISTICS:
- Channel Type: {patterns.channel_type.replace('_', ' ').title()}
- Content Style: {patterns.content_style.replace('_', ' ').title()}
- Data Confidence: {patterns.confidence_score:.1%}
"""
        
        # Add pattern-specific guidance
        pattern_guidance = self._generate_pattern_guidance(patterns)
        
        return base_prompt + channel_context + pattern_guidance
    
    def _customize_user_prompt(self, template: str, patterns: IntelligentPatterns) -> str:
        """Customize user prompt template based on patterns"""
        
        # Add adaptive emphasis based on strongest patterns
        emphasis_section = self._generate_emphasis_section(patterns)
        
        # Insert emphasis before the main instruction
        insertion_point = template.find("Please generate")
        if insertion_point != -1:
            return (template[:insertion_point] + emphasis_section + "\n\n" + 
                   template[insertion_point:])
        else:
            return template + "\n\n" + emphasis_section
    
    def _generate_pattern_guidance(self, patterns: IntelligentPatterns) -> str:
        """Generate specific guidance based on strongest patterns"""
        guidance = "\n\nOPTIMIZATION PRIORITIES:"
        
        weights = patterns.pattern_weights
        
        # Sort patterns by weight to prioritize
        pattern_priorities = [
            ("Word Count", weights.word_count_weight, f"Target ~{patterns.avg_word_count:.0f} words"),
            ("Questions", weights.question_weight, f"Use questions ({patterns.question_percentage:.0%} effective rate)"),
            ("Numbers", weights.numeric_weight, f"Include numbers ({patterns.numeric_percentage:.0%} effective rate)"),
            ("Exclamations", weights.exclamation_weight, f"Consider exclamations ({patterns.exclamation_percentage:.0%} effective rate)"),
            ("Capitalization", weights.capitalization_weight, f"Optimize capitalization ({patterns.capitalization_score:.1%} cap rate)"),
            ("Keywords", weights.keyword_weight, f"Use high-performing keywords: {', '.join(patterns.top_keywords[:5])}")
        ]
        
        # Sort by weight and take top 3
        pattern_priorities.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, weight, guidance_text) in enumerate(pattern_priorities[:3]):
            guidance += f"\n{i+1}. {name} (Weight: {weight:.2f}): {guidance_text}"
        
        return guidance
    
    def _generate_emphasis_section(self, patterns: IntelligentPatterns) -> str:
        """Generate emphasis section for user prompt"""
        
        emphasis = "CRITICAL SUCCESS FACTORS FOR THIS CHANNEL:"
        
        # Emphasize based on channel characteristics
        if patterns.content_style == "educational":
            emphasis += "\n- Focus on learning outcomes and practical value"
            emphasis += "\n- Use clear, benefit-driven language"
            
        elif patterns.content_style == "entertainment":
            emphasis += "\n- Prioritize emotional hooks and curiosity gaps"
            emphasis += "\n- Use engaging, energetic language"
            
        else:  # mixed
            emphasis += "\n- Balance educational value with entertainment appeal"
            emphasis += "\n- Use versatile, broadly appealing language"
        
        # Add confidence-based guidance
        if patterns.confidence_score < 0.5:
            emphasis += "\n- Apply general best practices due to limited data"
        elif patterns.confidence_score > 0.8:
            emphasis += "\n- Strictly follow discovered patterns (high confidence data)"
        
        return emphasis
    
    def _initialize_strategies(self) -> Dict[PromptStrategy, ContextualPrompt]:
        """Initialize all prompt strategies"""
        
        strategies = {}
        
        # High Volume Educational Strategy
        strategies[PromptStrategy.HIGH_VOLUME_EDUCATIONAL] = ContextualPrompt(
            system_prompt="""You are a YouTube title optimization expert specializing in educational content. 
            Create titles that maximize learning appeal and click-through rates for high-performing educational channels. 
            Focus on clear value propositions, learning outcomes, and authority-building language.""",
            
            user_prompt_template="""Based on this high-performing educational channel's patterns, generate {n_titles} optimized titles.
            
{channel_analysis}

NEW VIDEO IDEA: "{video_idea}"

Generate titles that emphasize learning value, practical benefits, and expertise. Each title should:
- Clearly communicate the learning outcome
- Use proven educational title patterns from this channel
- Appeal to viewers seeking knowledge and skills

Format each title with reasoning based on educational content patterns.""",
            
            parameters=AdaptiveParameters(
                temperature=0.6,  # Moderate creativity for educational content
                max_tokens=1200,
                top_p=0.9
            ),
            strategy=PromptStrategy.HIGH_VOLUME_EDUCATIONAL
        )
        
        # High Volume Entertainment Strategy
        strategies[PromptStrategy.HIGH_VOLUME_ENTERTAINMENT] = ContextualPrompt(
            system_prompt="""You are a YouTube title optimization expert specializing in entertainment content. 
            Create titles that maximize emotional engagement and viral potential for high-performing entertainment channels. 
            Focus on curiosity gaps, emotional hooks, and shareable content angles.""",
            
            user_prompt_template="""Based on this high-performing entertainment channel's patterns, generate {n_titles} engaging titles.
            
{channel_analysis}

NEW VIDEO IDEA: "{video_idea}"

Generate titles that maximize entertainment value and shareability. Each title should:
- Create strong curiosity gaps or emotional hooks
- Use proven entertainment patterns from this channel
- Appeal to viewers seeking fun, surprise, or emotional content

Format each title with reasoning based on entertainment content patterns.""",
            
            parameters=AdaptiveParameters(
                temperature=0.8,  # Higher creativity for entertainment
                max_tokens=1200,
                top_p=0.95
            ),
            strategy=PromptStrategy.HIGH_VOLUME_ENTERTAINMENT
        )
        
        # Medium Volume Mixed Strategy
        strategies[PromptStrategy.MEDIUM_VOLUME_MIXED] = ContextualPrompt(
            system_prompt="""You are a YouTube title optimization expert for diverse content channels. 
            Create titles that balance multiple content types and audience segments. 
            Focus on broad appeal while respecting discovered channel patterns.""",
            
            user_prompt_template="""Based on this diverse channel's performance patterns, generate {n_titles} versatile titles.
            
{channel_analysis}

NEW VIDEO IDEA: "{video_idea}"

Generate titles that appeal to this channel's diverse audience. Each title should:
- Balance entertainment and informational value
- Use the most effective patterns discovered for this channel
- Appeal to multiple viewer motivations

Format each title with reasoning based on mixed content patterns.""",
            
            parameters=AdaptiveParameters(
                temperature=0.7,  # Balanced creativity
                max_tokens=1000,
                top_p=0.9
            ),
            strategy=PromptStrategy.MEDIUM_VOLUME_MIXED
        )
        
        # Low Volume Adaptive Strategy
        strategies[PromptStrategy.LOW_VOLUME_ADAPTIVE] = ContextualPrompt(
            system_prompt="""You are a YouTube title optimization expert working with emerging channels. 
            Create titles using proven general best practices while incorporating available channel data. 
            Focus on broad appeal and tested title strategies.""",
            
            user_prompt_template="""Generate {n_titles} titles for this emerging channel, combining available data with best practices.
            
{channel_analysis}

NEW VIDEO IDEA: "{video_idea}"

Generate titles using both channel insights and proven YouTube strategies. Each title should:
- Apply general YouTube title best practices
- Incorporate any reliable patterns from available data
- Focus on broad audience appeal

Format each title with reasoning based on available data and general best practices.""",
            
            parameters=AdaptiveParameters(
                temperature=0.7,  # Moderate creativity with broader exploration
                max_tokens=900,
                top_p=0.85
            ),
            strategy=PromptStrategy.LOW_VOLUME_ADAPTIVE
        )
        
        # Default Strategy
        strategies[PromptStrategy.DEFAULT] = ContextualPrompt(
            system_prompt="""You are a YouTube title optimization expert. Create engaging, click-worthy titles 
            based on general best practices and any available channel data.""",
            
            user_prompt_template="""Generate {n_titles} optimized YouTube titles for the following video idea.
            
{channel_analysis}

VIDEO IDEA: "{video_idea}"

Create titles that are engaging, clear, and likely to perform well on YouTube. 
Format each title with brief reasoning.""",
            
            parameters=AdaptiveParameters(
                temperature=0.7,
                max_tokens=800,
                top_p=0.9
            ),
            strategy=PromptStrategy.DEFAULT
        )
        
        return strategies