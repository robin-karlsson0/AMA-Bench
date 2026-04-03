import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ModelClient:
    """Unified client for different model providers."""

    def __init__(self, config_path: str, server_type: str = "api", **kwargs):
        """
        Initialize model client.

        Args:
            config_path: Path to YAML config file containing provider, model, api_key, etc.
            server_type: Server type ("api" or "vllm")
            **kwargs: Additional provider-specific arguments (e.g., host, port for vllm)
        """
        # Load config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Extract provider and model from config
        self.provider = config.get('provider', '').lower()
        self.model = config.get('model', '')
        self.config = config
        # None means unset (don't send extra_body); True/False controls thinking mode
        self.enable_thinking: Optional[bool] = config.get(
            'enable_thinking', None)

        # For vllm server type, override provider
        if server_type == "vllm":
            self.provider = "vllm"
            # Use host/port from kwargs or config (support both 'host' and 'vllm_host' naming)
            kwargs.setdefault(
                'host',
                config.get('vllm_host') or config.get('host', 'localhost'))
            kwargs.setdefault(
                'port',
                config.get('vllm_port') or config.get('port', 8000))

        self.client = self._initialize_client(**kwargs)

    def _initialize_client(self, **kwargs):
        """Initialize provider-specific client."""
        if self.provider == "vllm":
            from openai import OpenAI
            base_url = f"http://{kwargs.get('host', 'localhost')}:{kwargs.get('port', 8000)}/v1"
            return OpenAI(base_url=base_url, api_key="EMPTY")

        elif self.provider == "openai":
            from openai import OpenAI

            # Use api_key from config if provided, otherwise will use OPENAI_API_KEY env var
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAI(api_key=api_key)
            else:
                # Let OpenAI SDK handle the API key (will use OPENAI_API_KEY env var)
                return OpenAI()

        elif self.provider == "deepseek":
            from openai import OpenAI
            api_key = self.config.get("api_key") or os.getenv(
                "DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek API key not found in config or DEEPSEEK_API_KEY environment variable"
                )
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not found in config or GOOGLE_API_KEY environment variable"
                )
            genai.configure(api_key=api_key)
            return genai

        elif self.provider in ["anthropic", "claude"]:
            from anthropic import Anthropic
            api_key = self.config.get("api_key") or os.getenv(
                "ANTHROPIC_API_KEY")
            if api_key:
                return Anthropic(api_key=api_key)
            else:
                # Let Anthropic SDK handle the API key (will use ANTHROPIC_API_KEY env var)
                return Anthropic()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def query(self,
              prompt: str,
              temperature: float = 0.0,
              max_tokens: int = 4096,
              max_retries: int = 5,
              system: Optional[str] = None,
              seed: Optional[int] = None) -> str:
        """Query model with prompt with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                if self.provider in ["vllm", "deepseek"]:
                    extra_body = None
                    if self.provider == "vllm" and self.enable_thinking is not None:
                        extra_body = {
                            "chat_template_kwargs": {
                                "enable_thinking": self.enable_thinking
                            }
                        }
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **({"seed": seed} if seed is not None else {}),
                        **(({
                            "extra_body": extra_body
                        }) if extra_body is not None else {}),
                    )
                    msg = response.choices[0].message
                    content = msg.content
                    if content is None:
                        # Reasoning models (e.g. Qwen3-thinking) separate think
                        # tokens into reasoning_content; content holds the final
                        # answer and may be None if max_tokens was exhausted by
                        # thinking.  Fall back to reasoning_content so the run
                        # does not crash, but warn so the user can increase
                        # max_tokens or disable thinking on the vLLM server.
                        content = getattr(msg, "reasoning_content", None) or ""
                        if not content:
                            print(
                                "⚠️  Warning: model returned empty content and "
                                "no reasoning_content. Consider increasing "
                                "max_response_len or disabling thinking mode.")
                    return content.strip()

                elif self.provider == "openai":
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }],
                            max_completion_tokens=max_tokens,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        if "max_completion_tokens" in str(
                                e) or "unsupported_parameter" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{
                                    "role": "user",
                                    "content": prompt
                                }],
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return response.choices[0].message.content.strip()
                        else:
                            raise

                elif self.provider == "gemini":
                    model = self.client.GenerativeModel(self.model)
                    response = model.generate_content(
                        prompt,
                        generation_config=self.client.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        ))
                    return response.text.strip()

                elif self.provider in ["anthropic", "claude"]:
                    # Build request parameters
                    request_params = {
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    # Add system parameter if provided
                    if system:
                        request_params["system"] = system

                    response = self.client.messages.create(**request_params)

                    # Handle response content
                    if not response.content:
                        # Check if it's a refusal
                        if hasattr(response, 'stop_reason'
                                   ) and response.stop_reason == 'refusal':
                            raise ValueError(
                                f"Claude refused to respond. This may be due to content policy. Stop reason: {response.stop_reason}"
                            )
                        raise ValueError(
                            f"Empty response content from Claude API. Response: {response}"
                        )
                    if not hasattr(response.content[0], 'text'):
                        raise ValueError(
                            f"Response content has no text attribute. Content type: {type(response.content[0])}, Content: {response.content[0]}"
                        )
                    return response.content[0].text.strip()

                else:
                    raise ValueError(
                        f"Query not implemented for provider: {self.provider}")

            except Exception as e:
                error_str = str(e)
                # Don't retry on refusals or permanent errors
                if "refused" in error_str.lower(
                ) or "refusal" in error_str.lower():
                    raise
                # Check if it's a rate limit error that should be retried
                if "rate" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        print(
                            f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                # For other errors, retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 10)  # Cap at 10 seconds
                    print(f"Error: {error_str}")
                    print(
                        f"Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")
