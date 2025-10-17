"""
Performance Tracking System with LangSmith Integration
Comprehensive logging and monitoring of LLM performance, prompts, and results
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from langsmith import Client
from .base_llm import LLMResponse


@dataclass
class PerformanceMetrics:
    """Performance metrics for LLM operations"""

    request_id: str
    timestamp: datetime
    model_name: str
    provider: str
    prompt_template: Optional[str]
    system_prompt: str
    user_prompt: str
    response_content: str
    response_time: float
    tokens_used: Optional[int]
    estimated_cost: Optional[float]
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PerformanceTracker:
    """
    Tracks LLM performance with LangSmith integration
    """

    def __init__(
        self, enable_file_logging: bool = True, log_file_path: Optional[str] = None
    ):
        """
        Initialize performance tracker

        Args:
            enable_file_logging: Whether to log to file
            log_file_path: Custom log file path
        """
        self.enable_file_logging = enable_file_logging
        self.langsmith_client = self._setup_langsmith()
        self.log_file_path = log_file_path or self._get_default_log_path()
        self._ensure_log_directory()

    def _setup_langsmith(self) -> Optional[Client]:
        """Setup LangSmith client if available"""
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            try:
                return Client(api_key=api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize LangSmith client: {e}")
        return None

    def _get_default_log_path(self) -> Path:
        """Get default log file path"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        return log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        if self.enable_file_logging:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def track_request(
        self,
        system_prompt: str,
        user_prompt: str,
        llm_response: LLMResponse,
        prompt_template: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Track a single LLM request

        Args:
            system_prompt: System prompt used
            user_prompt: User prompt used
            llm_response: Response from LLM
            prompt_template: Name of prompt template used
            additional_metadata: Additional metadata to track

        Returns:
            str: Request ID for tracking
        """
        request_id = str(uuid.uuid4())

        # Create performance metrics
        metrics = PerformanceMetrics(
            request_id=request_id,
            timestamp=datetime.now(),
            model_name=llm_response.model_used,
            provider=llm_response.provider,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_content=llm_response.content,
            response_time=llm_response.response_time or 0.0,
            tokens_used=llm_response.tokens_used,
            estimated_cost=llm_response.cost,
            success=True,
            metadata={**(llm_response.metadata or {}), **(additional_metadata or {})},
        )

        # Log to file
        if self.enable_file_logging:
            self._log_to_file(metrics)

        # Log to LangSmith
        if self.langsmith_client:
            self._log_to_langsmith(metrics)

        return request_id

    def track_error(
        self,
        system_prompt: str,
        user_prompt: str,
        error: Exception,
        model_name: str,
        provider: str,
        prompt_template: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Track a failed LLM request

        Args:
            system_prompt: System prompt used
            user_prompt: User prompt used
            error: Exception that occurred
            model_name: Model that was being used
            provider: Provider that was being used
            prompt_template: Name of prompt template used
            additional_metadata: Additional metadata to track

        Returns:
            str: Request ID for tracking
        """
        request_id = str(uuid.uuid4())

        # Create performance metrics for error
        metrics = PerformanceMetrics(
            request_id=request_id,
            timestamp=datetime.now(),
            model_name=model_name,
            provider=provider,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_content="",
            response_time=0.0,
            tokens_used=None,
            estimated_cost=None,
            success=False,
            error_message=str(error),
            metadata=additional_metadata or {},
        )

        # Log to file
        if self.enable_file_logging:
            self._log_to_file(metrics)

        # Log to LangSmith
        if self.langsmith_client:
            self._log_to_langsmith(metrics)

        return request_id

    def _log_to_file(self, metrics: PerformanceMetrics):
        """Log metrics to file in JSONL format"""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except Exception as e:
            print(f"Warning: Failed to log to file: {e}")

    def _log_to_langsmith(self, metrics: PerformanceMetrics):
        """Log metrics to LangSmith"""
        try:
            # Create a run in LangSmith
            run_data = {
                "name": f"title_generation_{metrics.model_name}",
                "run_type": "llm",
                "inputs": {
                    "system_prompt": metrics.system_prompt,
                    "user_prompt": metrics.user_prompt,
                    "prompt_template": metrics.prompt_template,
                },
                "outputs": (
                    {"response": metrics.response_content}
                    if metrics.success
                    else {"error": metrics.error_message}
                ),
                "metadata": {
                    "model_name": metrics.model_name,
                    "provider": metrics.provider,
                    "response_time": metrics.response_time,
                    "tokens_used": metrics.tokens_used,
                    "estimated_cost": metrics.estimated_cost,
                    "success": metrics.success,
                    **metrics.metadata,
                },
            }

            # Log the run
            self.langsmith_client.create_run(**run_data)

        except Exception as e:
            print(f"Warning: Failed to log to LangSmith: {e}")

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours

        Args:
            hours: Number of hours to look back

        Returns:
            Dict with performance summary
        """
        if not self.enable_file_logging:
            return {"error": "File logging not enabled"}

        try:
            metrics_list = self._load_recent_metrics(hours)

            if not metrics_list:
                return {"message": "No data found for the specified period"}

            # Calculate summary statistics
            total_requests = len(metrics_list)
            successful_requests = sum(1 for m in metrics_list if m["success"])
            failed_requests = total_requests - successful_requests

            response_times = [m["response_time"] for m in metrics_list if m["success"]]
            avg_response_time = (
                sum(response_times) / len(response_times) if response_times else 0
            )

            total_tokens = sum(
                m["tokens_used"] for m in metrics_list if m["tokens_used"]
            )
            total_cost = sum(
                m["estimated_cost"] for m in metrics_list if m["estimated_cost"]
            )

            # Group by model
            model_stats = {}
            for metrics in metrics_list:
                model = metrics["model_name"]
                if model not in model_stats:
                    model_stats[model] = {"count": 0, "success": 0, "total_time": 0}

                model_stats[model]["count"] += 1
                if metrics["success"]:
                    model_stats[model]["success"] += 1
                    model_stats[model]["total_time"] += metrics["response_time"]

            # Calculate model performance
            for model, stats in model_stats.items():
                stats["success_rate"] = stats["success"] / stats["count"]
                stats["avg_response_time"] = (
                    stats["total_time"] / stats["success"] if stats["success"] else 0
                )

            return {
                "period_hours": hours,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": (
                    successful_requests / total_requests if total_requests else 0
                ),
                "avg_response_time": avg_response_time,
                "total_tokens_used": total_tokens,
                "total_estimated_cost": total_cost,
                "model_performance": model_stats,
                "langsmith_enabled": self.langsmith_client is not None,
            }

        except Exception as e:
            return {"error": f"Failed to generate summary: {str(e)}"}

    def _load_recent_metrics(self, hours: int) -> List[Dict[str, Any]]:
        """Load metrics from the last N hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        metrics_list = []

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(
                            data["timestamp"]
                        ).timestamp()
                        if timestamp >= cutoff_time:
                            metrics_list.append(data)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            pass

        return metrics_list

    def get_model_comparison(self, hours: int = 24) -> Dict[str, Any]:
        """Get comparison of different models' performance"""
        try:
            metrics_list = self._load_recent_metrics(hours)

            if not metrics_list:
                return {"message": "No data found for comparison"}

            model_comparison = {}

            for metrics in metrics_list:
                model = metrics["model_name"]
                provider = metrics["provider"]
                key = f"{provider}_{model}"

                if key not in model_comparison:
                    model_comparison[key] = {
                        "model": model,
                        "provider": provider,
                        "total_requests": 0,
                        "successful_requests": 0,
                        "response_times": [],
                        "token_counts": [],
                        "costs": [],
                    }

                model_comparison[key]["total_requests"] += 1

                if metrics["success"]:
                    model_comparison[key]["successful_requests"] += 1
                    model_comparison[key]["response_times"].append(
                        metrics["response_time"]
                    )

                    if metrics["tokens_used"]:
                        model_comparison[key]["token_counts"].append(
                            metrics["tokens_used"]
                        )

                    if metrics["estimated_cost"]:
                        model_comparison[key]["costs"].append(metrics["estimated_cost"])

            # Calculate statistics for each model
            for key, data in model_comparison.items():
                data["success_rate"] = (
                    data["successful_requests"] / data["total_requests"]
                )
                data["avg_response_time"] = (
                    sum(data["response_times"]) / len(data["response_times"])
                    if data["response_times"]
                    else 0
                )
                data["avg_tokens"] = (
                    sum(data["token_counts"]) / len(data["token_counts"])
                    if data["token_counts"]
                    else 0
                )
                data["avg_cost"] = (
                    sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0
                )
                data["total_cost"] = sum(data["costs"])

                # Remove raw lists to keep response clean
                del data["response_times"]
                del data["token_counts"]
                del data["costs"]

            return {
                "period_hours": hours,
                "model_comparison": model_comparison,
                "total_models": len(model_comparison),
            }

        except Exception as e:
            return {"error": f"Failed to generate model comparison: {str(e)}"}


# Global instance
performance_tracker = PerformanceTracker()
