"""
Title Generator

Orchestrates data loading, LLM service, prompt management, and performance tracking
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

from src.data_module.data_processor import DataLoader, GeneratedTitle
from .llm_config import config_manager
from .llm_service import create_system_message, create_human_message, LLMGenerationError
from .prompt_manager import prompt_manager, create_title_generation_variables
from .performance_tracker import performance_tracker

@dataclass
class TitleGenerationRequest:
    """Request object for title generation"""

    channel_id: str
    video_idea: str
    n_titles: int = 4
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class TitleGenerationResponse:
    """Response object for title generation"""

    titles: List[GeneratedTitle]
    request_id: str
    model_used: str
    provider: str
    response_time: float
    tokens_used: Optional[int]
    estimated_cost: Optional[float]
    success: bool
    error_message: Optional[str] = None


class TitleGenerator:
    """Title generator with LLM integration and performance tracking"""

    def __init__(self, default_model: Optional[str] = None):
        """Initialize the title generator"""
        self.data_loader = DataLoader()
        self.default_model = default_model or "DeepSeek-R1-Distill-Qwen-32B"

    def generate_titles(self, request: TitleGenerationRequest) -> TitleGenerationResponse:
        """Generate titles for a video idea based on channel patterns"""
        start_time = time.time()

        try:
            return self._generate_titles_success(request, start_time)
        except Exception as e:
            return self._generate_titles_error(request, e, start_time)

    def _generate_titles_success(self, request: TitleGenerationRequest, start_time: float) -> TitleGenerationResponse:
        """Handle successful title generation"""
        model_name = request.model_name or self.default_model
        llm_service = config_manager.create_llm_service(model_name)
        
        # Prepare data and prompts
        messages = self._prepare_messages(request)
        generation_kwargs = self._prepare_generation_kwargs(request)
        
        # Generate and parse response
        llm_response = llm_service.generate(messages, **generation_kwargs)
        generated_titles = self._parse_llm_response(llm_response.content)
        
        # Track and return results
        response_time = time.time() - start_time
        request_id = performance_tracker.track_request(
            system_prompt=messages[0].content,
            user_prompt=messages[1].content,
            llm_response=llm_response,
            prompt_template="title_generation",
            additional_metadata={
                "channel_id": request.channel_id,
                "video_idea": request.video_idea,
                "n_titles": request.n_titles
            }
        )
        
        return TitleGenerationResponse(
            titles=generated_titles[:request.n_titles],
            request_id=request_id,
            model_used=model_name,
            provider=llm_service.config.provider.value,
            response_time=response_time,
            tokens_used=llm_response.tokens_used,
            estimated_cost=llm_response.cost,
            success=True,
        )

    def _generate_titles_error(self, request: TitleGenerationRequest, error: Exception, start_time: float) -> TitleGenerationResponse:
        """Handle title generation errors with fallback"""
        try:
            request_id = performance_tracker.track_error(
                system_prompt="",
                user_prompt="",
                error=error,
                model_name="unknown",
                provider="unknown",
                prompt_template="title_generation",
                additional_metadata={
                    "channel_id": request.channel_id,
                    "video_idea": request.video_idea,
                    "n_titles": request.n_titles
                }
            )
        except Exception:
            request_id = "fallback"

        fallback_titles = self._generate_fallback_titles(request.video_idea, request.n_titles)
        
        return TitleGenerationResponse(
            titles=fallback_titles,
            request_id=request_id,
            model_used="unknown",
            provider="fallback",
            response_time=time.time() - start_time,
            tokens_used=None,
            estimated_cost=None,
            success=False,
            error_message=str(error),
        )

    def _prepare_messages(self, request: TitleGenerationRequest):
        """Prepare messages for LLM generation"""
        channel_analysis = self.data_loader.analyze_channel(request.channel_id)
        
        prompt_variables = create_title_generation_variables(
            channel_id=request.channel_id,
            video_idea=request.video_idea,
            channel_analysis=channel_analysis,
            n_titles=request.n_titles,
        )
        
        system_prompt, user_prompt = prompt_manager.format_prompt(
            template_name="title_generation",
            variables=prompt_variables
        )
        
        return [
            create_system_message(system_prompt),
            create_human_message(user_prompt),
        ]

    def _prepare_generation_kwargs(self, request: TitleGenerationRequest) -> dict:
        """Prepare generation parameters"""
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        return kwargs

    def _parse_llm_response(self, response_text: str) -> List[GeneratedTitle]:
        """Parse LLM response into GeneratedTitle objects"""
        # Clean the response text first
        cleaned_response = self._clean_response_text(response_text)
        
        try:
            return self._parse_structured_response(cleaned_response)
        except Exception:
            return self._simple_parse_response(cleaned_response)

    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text by removing model internal reasoning"""
        import re
        
        # Remove <think>...</think> blocks (DeepSeek model internal reasoning)
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<reasoning>.*?</reasoning>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<internal>.*?</internal>', '', response_text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        response_text = re.sub(r'\n\s*\n', '\n\n', response_text)
        return response_text.strip()

    def _parse_structured_response(self, response_text: str) -> List[GeneratedTitle]:
        """Parse structured LLM response"""
        titles = []
        lines = response_text.strip().split("\n")
        current_title = None
        current_reasoning = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if self._is_title_line(line):
                if current_title:
                    titles.append(GeneratedTitle(title=current_title, reasoning=current_reasoning.strip()))
                current_title = self._extract_title_text(line)
                current_reasoning = ""
            elif self._is_reasoning_line(line):
                current_reasoning = self._extract_reasoning_text(line)
            elif current_title and self._is_continuation_line(line):
                current_reasoning += " " + line

        if current_title:
            titles.append(GeneratedTitle(title=current_title, reasoning=current_reasoning.strip()))

        return titles

    def _is_continuation_line(self, line: str) -> bool:
        """Check if line is continuation of reasoning"""
        return not self._is_title_line(line) and not line.startswith("---") and not line.startswith("**")

    def _is_title_line(self, line: str) -> bool:
        """Check if line contains a title"""
        return line.startswith("TITLE") or line.startswith("**TITLE")

    def _is_reasoning_line(self, line: str) -> bool:
        """Check if line contains reasoning"""
        return line.startswith("REASONING") or line.startswith("**REASONING")

    def _extract_title_text(self, line: str) -> str:
        """Extract title text from line"""
        title_part = line.split(":", 1)
        if len(title_part) > 1:
            title_text = title_part[1].strip()
            title_text = title_text.strip('"').strip("'")
            title_text = title_text.replace("**", "").strip()
            return title_text
        return ""

    def _extract_reasoning_text(self, line: str) -> str:
        """Extract reasoning text from line"""
        reasoning_part = line.split(":", 1)
        if len(reasoning_part) > 1:
            reasoning_text = reasoning_part[1].strip()
            reasoning_text = reasoning_text.replace("**", "").strip()
            return reasoning_text
        return ""

    def _simple_parse_response(self, response_text: str) -> List[GeneratedTitle]:
        """Simple fallback parsing for when structured parsing fails"""
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        titles = []

        for line in lines:
            if self._is_potential_title(line):
                clean_title = self._clean_title_line(line)
                if clean_title and len(clean_title) > 5:
                    titles.append(GeneratedTitle(title=clean_title, reasoning="Generated based on channel patterns"))

        # If no titles found, take first few lines as titles
        if not titles:
            for line in lines[:4]:
                if len(line) > 5:
                    titles.append(GeneratedTitle(title=line, reasoning="Generated using LLM"))

        return titles

    def _is_potential_title(self, line: str) -> bool:
        """Check if line could be a title"""
        markers = ["title", "•", "-", "1.", "2.", "3.", "4."]
        return any(marker in line.lower() for marker in markers)

    def _clean_title_line(self, line: str) -> str:
        """Clean title line to extract just the title"""
        clean_title = line
        markers = ["TITLE 1:", "TITLE 2:", "TITLE 3:", "TITLE 4:", "TITLE:", "•", "-"]
        for marker in markers:
            clean_title = clean_title.replace(marker, "").strip()
        return clean_title

    def _generate_fallback_titles(
        self, idea: str, n_titles: int
    ) -> List[GeneratedTitle]:
        """Generate simple fallback titles when LLM fails"""
        fallback_patterns = [
            f"How to {idea}",
            f"The Ultimate Guide to {idea}",
            f"Everything You Need to Know About {idea}",
            f"Amazing {idea} Tips That Actually Work",
            f"Why {idea} is Important",
            f"The Truth About {idea}",
            f"Incredible {idea} Facts",
            f"Master {idea} in Minutes",
        ]

        titles = []
        for i, pattern in enumerate(fallback_patterns[:n_titles]):
            titles.append(
                GeneratedTitle(
                    title=pattern,
                    reasoning="Fallback title generated using pattern templates",
                )
            )

        return titles

    def get_available_models(self) -> Dict[str, Any]:
        """Get available models from config manager"""
        return config_manager.get_available_models()


# For backward compatibility, maintain the old class name as an alias
EnhancedTitleGenerator = TitleGenerator