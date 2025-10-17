"""
Prompt Manager with Python f-string Templates
Handles prompt templates with placeholders for dynamic content injection
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts available"""

    TITLE_GENERATION = "title_generation"
    CONTENT_ANALYSIS = "content_analysis"
    PATTERN_ANALYSIS = "pattern_analysis"


@dataclass
class PromptTemplate:
    """Template for prompts with metadata"""

    name: str
    prompt_type: PromptType
    system_prompt: str
    user_prompt: str
    description: str
    required_variables: List[str]
    optional_variables: Optional[List[str]] = None

    def __post_init__(self):
        if self.optional_variables is None:
            self.optional_variables = []


class PromptManager:
    """
    Manages prompt templates with f-string formatting
    """

    def __init__(self):
        """Initialize with predefined prompt templates"""
        self.templates = self._load_default_templates()

    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default prompt templates"""
        templates = {}

        # Title Generation Template
        templates["title_generation"] = PromptTemplate(
            name="title_generation",
            prompt_type=PromptType.TITLE_GENERATION,
            system_prompt="""You are a YouTube title optimization expert. Create engaging titles based on successful patterns from the channel's history. Focus on maximizing click-through rates while maintaining authenticity to the channel's style.""",
            user_prompt="""Based on this YouTube channel's most successful titles and patterns, generate {n_titles} engaging titles for a new video.

CHANNEL PERFORMANCE DATA:
- Channel ID: {channel_id}
- Total videos: {total_videos}
- Average views: {avg_views:,.0f}

TOP PERFORMING TITLES:
{top_titles}

SUCCESSFUL PATTERNS IDENTIFIED:
- Average word count: {avg_word_count:.1f} words
- Question titles: {question_percentage:.0%} of top performers
- Titles with numbers: {numeric_percentage:.0%} of top performers  
- Exclamation usage: {exclamation_percentage:.0%} of top performers
- Common successful words: {common_words}

NEW VIDEO IDEA: "{video_idea}"

Please generate {n_titles} titles that follow the successful patterns from this channel. For each title, provide:
1. The title text
2. Brief reasoning explaining which patterns it uses and why it should perform well

Format as:
TITLE 1: [title text]
REASONING: [explanation of patterns used and expected performance]

TITLE 2: [title text] 
REASONING: [explanation of patterns used and expected performance]

Continue for all {n_titles} titles.""",
            description="Generate YouTube titles based on channel analysis and video idea",
            required_variables=[
                "channel_id",
                "video_idea",
                "n_titles",
                "total_videos",
                "avg_views",
                "top_titles",
                "avg_word_count",
                "question_percentage",
                "numeric_percentage",
                "exclamation_percentage",
                "common_words",
            ],
        )

        # Content Analysis Template
        templates["content_analysis"] = PromptTemplate(
            name="content_analysis",
            prompt_type=PromptType.CONTENT_ANALYSIS,
            system_prompt="""You are a content analysis expert. Analyze the provided content and extract key insights, patterns, and recommendations.""",
            user_prompt="""Analyze the following content and provide insights:

CONTENT TYPE: {content_type}
CONTENT DATA:
{content_data}

Please provide:
1. Key themes and topics
2. Performance patterns
3. Audience engagement indicators
4. Content optimization recommendations

Analysis for: {analysis_purpose}""",
            description="Analyze content for patterns and insights",
            required_variables=["content_type", "content_data", "analysis_purpose"],
        )

        # Pattern Analysis Template
        templates["pattern_analysis"] = PromptTemplate(
            name="pattern_analysis",
            prompt_type=PromptType.PATTERN_ANALYSIS,
            system_prompt="""You are a data pattern analysis expert. Identify trends, patterns, and actionable insights from the provided data.""",
            user_prompt="""Analyze the following data patterns:

DATA SUMMARY:
{data_summary}

METRICS:
{metrics}

TIME PERIOD: {time_period}

Please identify:
1. Key trends and patterns
2. Performance drivers
3. Optimization opportunities
4. Predictions and recommendations

Focus on: {focus_area}""",
            description="Analyze data patterns and trends",
            required_variables=["data_summary", "metrics", "time_period", "focus_area"],
        )

        return templates

    def get_template(self, template_name: str) -> PromptTemplate:
        """
        Get a prompt template by name

        Args:
            template_name: Name of the template

        Returns:
            PromptTemplate: The requested template
        """
        if template_name not in self.templates:
            raise ValueError(
                f"Template '{template_name}' not found. Available: {list(self.templates.keys())}"
            )

        return self.templates[template_name]

    def format_prompt(
        self, template_name: str, variables: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Format a prompt template with provided variables

        Args:
            template_name: Name of the template to use
            variables: Dictionary of variables to substitute

        Returns:
            tuple: (system_prompt, user_prompt) formatted with variables
        """
        template = self.get_template(template_name)

        # Validate required variables
        missing_vars = []
        for var in template.required_variables:
            if var not in variables:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(
                f"Missing required variables for template '{template_name}': {missing_vars}"
            )

        # Add default values for optional variables if not provided
        complete_variables = variables.copy()
        for var in template.optional_variables:
            if var not in complete_variables:
                complete_variables[var] = ""

        try:
            # Format prompts with f-string style
            system_prompt = template.system_prompt.format(**complete_variables)
            user_prompt = template.user_prompt.format(**complete_variables)

            return system_prompt, user_prompt

        except KeyError as e:
            raise ValueError(
                f"Variable {e} not provided for template '{template_name}'"
            )
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {str(e)}")

    def add_template(self, template: PromptTemplate):
        """Add a custom prompt template"""
        self.templates[template.name] = template

    def get_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        template = self.get_template(template_name)
        return {
            "name": template.name,
            "type": template.prompt_type.value,
            "description": template.description,
            "required_variables": template.required_variables,
            "optional_variables": template.optional_variables,
        }

    def validate_variables(
        self, template_name: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate variables against template requirements

        Returns:
            Dict with validation results
        """
        template = self.get_template(template_name)

        missing_required = []
        for var in template.required_variables:
            if var not in variables:
                missing_required.append(var)

        extra_variables = []
        all_template_vars = set(
            template.required_variables + template.optional_variables
        )
        for var in variables:
            if var not in all_template_vars:
                extra_variables.append(var)

        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "extra_variables": extra_variables,
            "template_name": template_name,
        }


# Convenience functions for common prompt patterns
def create_title_generation_variables(
    channel_id: str, video_idea: str, channel_analysis, n_titles: int = 4
) -> Dict[str, Any]:
    """
    Create variables dictionary for title generation prompt

    Args:
        channel_id: YouTube channel ID
        video_idea: Idea for the new video
        channel_analysis: Analysis object with channel data
        n_titles: Number of titles to generate

    Returns:
        Dict with all required variables for title generation
    """
    # Format top titles with view counts
    top_titles_formatted = []
    for i, video in enumerate(channel_analysis.top_performers[:5], 1):
        top_titles_formatted.append(
            f"{i}. {video.title} ({video.views_in_period:,} views)"
        )

    top_titles_text = "\n".join(top_titles_formatted)

    # Format common words
    common_words_text = ", ".join(
        [word for word, count in channel_analysis.patterns.get("common_words", [])[:10]]
    )

    return {
        "channel_id": channel_id,
        "video_idea": video_idea,
        "n_titles": n_titles,
        "total_videos": channel_analysis.total_videos,
        "avg_views": channel_analysis.avg_views,
        "top_titles": top_titles_text,
        "avg_word_count": channel_analysis.patterns.get("avg_length", 0),
        "question_percentage": channel_analysis.patterns.get("question_titles", 0),
        "numeric_percentage": channel_analysis.patterns.get("numeric_titles", 0),
        "exclamation_percentage": channel_analysis.patterns.get(
            "exclamation_titles", 0
        ),
        "common_words": common_words_text,
    }


# Global instance
prompt_manager = PromptManager()
