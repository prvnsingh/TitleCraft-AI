"""
Enhanced Title Generator using LangChain LLM Service
Integrates with the new plug-and-play LLM service for better flexibility
"""
from typing import List, Dict, Any, Optional
import json
import os

try:
    from src.services.llm import (
        LLMServiceFactory, 
        create_system_message, 
        create_human_message,
        get_default_service
    )
    from src.services.llm_config import get_recommended_config, validate_config
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available, falling back to direct OpenAI integration")

from src.data_module.data_processor import DataLoader, ChannelAnalysis, GeneratedTitle
from src.services.title_generator import TitleGenerator as OriginalTitleGenerator


class EnhancedTitleGenerator:
    """Enhanced title generator using LangChain LLM service with tracing"""
    
    def __init__(self, llm_provider: str = "openai", model: str = None, api_key: str = None):
        self.data_loader = DataLoader()
        
        # Initialize LLM service with fallback
        if LANGCHAIN_AVAILABLE:
            self.llm_service = self._initialize_llm_service(llm_provider, model, api_key)
            self.use_langchain = True
        else:
            # Fallback to original implementation
            self.original_generator = OriginalTitleGenerator(api_key)
            self.use_langchain = False
    
    def _initialize_llm_service(self, provider: str, model: str, api_key: str):
        """Initialize the LLM service based on configuration"""
        try:
            # Get recommended configuration for title generation
            config = get_recommended_config("title_generation")
            
            # Override with provided parameters
            if provider:
                config["provider"] = provider
            if model:
                config["model"] = model
            
            # Create service based on provider
            if config["provider"] == "openai":
                return LLMServiceFactory.create_openai_service(
                    model=config["model"],
                    api_key=api_key,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"]
                )
            elif config["provider"] == "anthropic":
                return LLMServiceFactory.create_anthropic_service(
                    model=config["model"],
                    api_key=api_key,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"]
                )
            elif config["provider"] == "ollama":
                return LLMServiceFactory.create_ollama_service(
                    model=config["model"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"]
                )
            else:
                raise ValueError(f"Unsupported provider: {config['provider']}")
                
        except Exception as e:
            print(f"Failed to initialize LLM service: {e}")
            # Fallback to default OpenAI service
            return get_default_service("openai_fast")
    
    def generate_titles(self, channel_id: str, idea: str, n_titles: int = 4) -> List[GeneratedTitle]:
        """Generate titles using the LangChain LLM service"""
        if not self.use_langchain:
            # Fallback to original generator
            return self.original_generator.generate_titles(channel_id, idea, n_titles)
        
        try:
            # Analyze channel patterns
            channel_analysis = self.data_loader.analyze_channel(channel_id)
            
            # Create messages for LangChain
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(channel_analysis, idea, n_titles)
            
            messages = [
                create_system_message(system_prompt),
                create_human_message(user_prompt)
            ]
            
            # Generate response using LangChain service
            response = self.llm_service.generate(messages)
            
            # Parse and return titles
            generated_titles = self._parse_langchain_response(response, channel_analysis)
            
            return generated_titles[:n_titles]
            
        except Exception as e:
            print(f"LangChain generation failed: {e}")
            # Fallback to pattern-based titles
            return self._generate_fallback_titles(idea, n_titles)
    
    def generate_titles_streaming(self, channel_id: str, idea: str, n_titles: int = 4):
        """Generate titles with streaming response"""
        if not self.use_langchain:
            # Non-streaming fallback
            titles = self.original_generator.generate_titles(channel_id, idea, n_titles)
            for title in titles:
                yield title
            return
        
        try:
            # Analyze channel patterns
            channel_analysis = self.data_loader.analyze_channel(channel_id)
            
            # Create messages
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(channel_analysis, idea, n_titles)
            
            messages = [
                create_system_message(system_prompt),
                create_human_message(user_prompt)
            ]
            
            # Stream response
            full_response = ""
            for chunk in self.llm_service.generate_stream(messages):
                full_response += chunk
                yield {"chunk": chunk, "complete": False}
            
            # Parse final response
            titles = self._parse_langchain_response(full_response, channel_analysis)
            yield {"titles": titles[:n_titles], "complete": True}
            
        except Exception as e:
            yield {"error": str(e), "complete": True}
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LangChain"""
        return """You are a YouTube title optimization expert with deep expertise in content strategy and audience engagement.

Your task is to create compelling, data-driven YouTube titles that:
1. Follow successful patterns from the channel's historical data
2. Maximize click-through rates and audience retention
3. Are authentic to the channel's voice and content style
4. Include relevant keywords and emotional triggers
5. Maintain optimal length (typically 40-70 characters)

Analyze the provided channel data carefully and create titles that would genuinely perform well based on the evidence from successful videos on this specific channel."""
    
    def _create_user_prompt(self, channel_analysis: ChannelAnalysis, idea: str, n_titles: int) -> str:
        """Create user prompt with channel analysis and video idea"""
        # Get top patterns for prompt
        top_words = list(channel_analysis.word_frequency.keys())[:10]
        top_patterns = list(channel_analysis.title_patterns.keys())[:5]
        
        prompt = f"""
Channel Analysis Data:
- Channel ID: {channel_analysis.channel_id}
- Total videos analyzed: {channel_analysis.total_videos}
- Average title length: {channel_analysis.avg_title_length}
- Top performing words: {', '.join(top_words)}
- Successful title patterns: {', '.join(top_patterns)}
- Performance insights: {channel_analysis.performance_insights}

Video Idea: "{idea}"

Please generate {n_titles} optimized YouTube titles for this video idea based on the channel's successful patterns. 

For each title, provide:
1. The title itself (40-70 characters recommended)
2. Brief reasoning explaining how it follows the channel's successful patterns

Format your response as a JSON array with this structure:
[
  {{
    "title": "Generated Title Here",
    "reasoning": "Explanation of why this title should perform well based on channel data"
  }},
  ...
]

Focus on titles that authentically match this channel's proven success patterns while being engaging for the target audience."""
        
        return prompt
    
    def _parse_langchain_response(self, response: str, channel_analysis: ChannelAnalysis) -> List[GeneratedTitle]:
        """Parse LangChain response into GeneratedTitle objects"""
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                titles_data = json.loads(json_str)
                
                generated_titles = []
                for item in titles_data:
                    if isinstance(item, dict) and "title" in item and "reasoning" in item:
                        title_obj = GeneratedTitle(
                            title=item["title"],
                            reasoning=item["reasoning"],
                            confidence_score=0.8,  # Default confidence for LangChain generated
                            pattern_matches=[]  # Could be enhanced to extract actual patterns
                        )
                        generated_titles.append(title_obj)
                
                return generated_titles
            
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing if JSON parsing fails
        return self._parse_fallback_response(response, channel_analysis)
    
    def _parse_fallback_response(self, response: str, channel_analysis: ChannelAnalysis) -> List[GeneratedTitle]:
        """Fallback parsing for non-JSON responses"""
        lines = response.strip().split('\n')
        titles = []
        
        current_title = None
        current_reasoning = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered titles or quoted titles
            if any(char.isdigit() for char in line[:5]) or line.startswith('"'):
                # Extract title
                if '"' in line:
                    start = line.find('"')
                    end = line.rfind('"')
                    if start < end:
                        current_title = line[start+1:end]
                else:
                    # Try to extract after number and punctuation
                    for i, char in enumerate(line):
                        if char.isalpha():
                            current_title = line[i:].strip()
                            break
            
            elif current_title and ("reasoning" in line.lower() or "because" in line.lower()):
                current_reasoning = line
            
            elif current_title and current_reasoning:
                titles.append(GeneratedTitle(
                    title=current_title,
                    reasoning=current_reasoning,
                    confidence_score=0.7,
                    pattern_matches=[]
                ))
                current_title = None
                current_reasoning = None
        
        # Add last title if we have one
        if current_title:
            reasoning = current_reasoning or "Generated based on channel patterns"
            titles.append(GeneratedTitle(
                title=current_title,
                reasoning=reasoning,
                confidence_score=0.7,
                pattern_matches=[]
            ))
        
        return titles
    
    def _generate_fallback_titles(self, idea: str, n_titles: int) -> List[GeneratedTitle]:
        """Generate simple fallback titles"""
        fallback_titles = [
            f"The Ultimate Guide to {idea}",
            f"How to {idea} - Complete Tutorial", 
            f"Everything You Need to Know About {idea}",
            f"{idea} Explained in Simple Terms",
            f"The Truth About {idea}"
        ]
        
        titles = []
        for i, title in enumerate(fallback_titles[:n_titles]):
            titles.append(GeneratedTitle(
                title=title,
                reasoning="Fallback title using common successful patterns",
                confidence_score=0.5,
                pattern_matches=["tutorial", "guide"]
            ))
        
        return titles
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the current LLM service"""
        if self.use_langchain:
            service_info = self.llm_service.get_provider_info()
            service_info["langchain_enabled"] = True
            service_info["config_status"] = validate_config()
            return service_info
        else:
            return {
                "langchain_enabled": False,
                "provider": "openai",
                "model": "gpt-3.5-turbo", 
                "fallback_mode": True
            }
    
    def switch_provider(self, provider: str, model: str = None, api_key: str = None):
        """Switch to a different LLM provider"""
        if not self.use_langchain:
            print("Cannot switch provider: LangChain not available")
            return False
        
        try:
            self.llm_service = self._initialize_llm_service(provider, model, api_key)
            print(f"Successfully switched to {provider}")
            return True
        except Exception as e:
            print(f"Failed to switch provider: {e}")
            return False


# Backwards compatibility wrapper
class TitleGenerator(EnhancedTitleGenerator):
    """Backwards compatible title generator that uses the enhanced version"""
    
    def __init__(self, api_key: str = None):
        # Try to use enhanced version, fallback to original if needed
        super().__init__(llm_provider="openai", api_key=api_key)


# Example usage
if __name__ == "__main__":
    # Test the enhanced title generator
    generator = EnhancedTitleGenerator()
    
    # Print service info
    info = generator.get_service_info()
    print("Service Info:", json.dumps(info, indent=2))
    
    # Example title generation (would need actual data)
    # titles = generator.generate_titles("UC_test", "How to learn Python programming")
    # for title in titles:
    #     print(f"Title: {title.title}")
    #     print(f"Reasoning: {title.reasoning}")
    #     print("---")