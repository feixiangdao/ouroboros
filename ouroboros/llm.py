"""
Ouroboros — LLM client.

Supports two API formats controlled by OUROBOROS_API_TYPE:
  - "openai"             → OpenAI-compatible SDK  (/chat/completions)
  - "anthropic-messages" → Anthropic Messages SDK (/messages)

Configuration (env vars):
  OUROBOROS_BASE_URL  — API base URL     (default: https://anyrouter.top)
  OUROBOROS_API_KEY   — API key          (fallback: OPENROUTER_API_KEY)
  OUROBOROS_API_TYPE  — API format       (default: anthropic-messages)
  OUROBOROS_MODEL     — main model       (default: anyrouter/claude-opus-4-6)
  OUROBOROS_MODEL_CODE— code model       (default: same as OUROBOROS_MODEL)
  OUROBOROS_MODEL_LIGHT— light model     (default: anyrouter/claude-opus-4-6)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "anyrouter/claude-opus-4-6"

# providers.json lives at the repo root (two levels up from this file)
_PROVIDERS_FILE = pathlib.Path(__file__).parent.parent / "providers.json"


def list_providers() -> Dict[str, Any]:
    """Return all configured providers from providers.json. Empty dict on failure."""
    try:
        if _PROVIDERS_FILE.exists():
            return json.loads(_PROVIDERS_FILE.read_text(encoding="utf-8")).get("providers", {})
    except Exception as e:
        log.warning("Failed to load providers.json: %s", e)
    return {}


def list_available_models(provider: str = None) -> Dict[str, List[str]]:
    """Return available models grouped by provider.

    If *provider* is given, return only that provider's models.
    Returns ``{provider_name: [model_id, ...]}``.
    """
    providers = list_providers()
    result: Dict[str, List[str]] = {}
    for pname, pcfg in providers.items():
        if provider and pname != provider:
            continue
        result[pname] = list(pcfg.get("available_models") or [])
    return result


def apply_provider(name: str) -> bool:
    """
    Apply a named provider's config to env vars so all subsequent LLMClient
    instances (and newly spawned workers) use it.

    Returns True if the provider was found and applied, False otherwise.
    """
    providers = list_providers()
    cfg = providers.get(name)
    if not cfg:
        return False

    if cfg.get("base_url"):
        os.environ["OUROBOROS_BASE_URL"] = cfg["base_url"]

    api_key_env = cfg.get("api_key_env", "")
    if api_key_env:
        key = os.environ.get(api_key_env, "")
        if key:
            os.environ["OUROBOROS_API_KEY"] = key
            os.environ["OPENROUTER_API_KEY"] = key  # backward compat

    models = cfg.get("models", {})
    if models.get("main"):
        os.environ["OUROBOROS_MODEL"] = models["main"]
    if models.get("code"):
        os.environ["OUROBOROS_MODEL_CODE"] = models["code"]
    if models.get("light"):
        os.environ["OUROBOROS_MODEL_LIGHT"] = models["light"]

    api_type = cfg.get("api", "openai")
    os.environ["OUROBOROS_API_TYPE"] = api_type
    os.environ["OUROBOROS_PROVIDER"] = name
    log.info("Provider switched to '%s' (base_url=%s, api=%s, model=%s)",
             name, cfg.get("base_url"), api_type, models.get("main"))
    return True


# ---------------------------------------------------------------------------
# Message format converters  (OpenAI ↔ Anthropic)
# ---------------------------------------------------------------------------

def _msgs_to_anthropic(messages: List[Dict[str, Any]]) -> tuple:
    """Convert OpenAI-format messages to Anthropic format.
    Returns (system_prompt: str, anthropic_messages: list).
    """
    system = ""
    result: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            system = str(content or "")
            continue

        if role == "tool":
            # Tool result → user message with tool_result block
            block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": str(content or ""),
            }
            # Merge consecutive tool results into one user message
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                result[-1]["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:8]}",
                    "name": fn.get("name", ""),
                    "input": inp,
                })
            if blocks:
                result.append({"role": "assistant", "content": blocks})
            continue

        if role == "user":
            if isinstance(content, list):
                result.append({"role": "user", "content": content})
            else:
                result.append({"role": "user", "content": str(content or "")})

    return system, result


def _tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI function-tool defs to Anthropic tool format."""
    out = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def _anthropic_resp_to_openai(response: Any) -> tuple:
    """Convert Anthropic response object to OpenAI-style (msg_dict, usage_dict)."""
    msg: Dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": None}
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in (getattr(response, "content", None) or []):
        btype = getattr(block, "type", "")
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input or {}),
                },
            })

    if text_parts:
        msg["content"] = "\n".join(text_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls

    u = getattr(response, "usage", None) or {}
    input_tok  = getattr(u, "input_tokens",  0) or 0
    output_tok = getattr(u, "output_tokens", 0) or 0
    cached     = getattr(u, "cache_read_input_tokens",    0) or 0
    cache_wri  = getattr(u, "cache_creation_input_tokens", 0) or 0

    usage: Dict[str, Any] = {
        "prompt_tokens":     input_tok,
        "completion_tokens": output_tok,
        "total_tokens":      input_tok + output_tok,
        "cached_tokens":     cached,
        "cache_write_tokens": cache_wri,
        "cost": None,
    }
    return msg, usage


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper. Supports OpenAI-compatible and Anthropic Messages formats."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_type: Optional[str] = None,
    ):
        self._api_key = (
            api_key
            or os.environ.get("OUROBOROS_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        self._base_url = (
            base_url
            or os.environ.get("OUROBOROS_BASE_URL", "https://anyrouter.top")
        )
        self._api_type = (
            api_type
            or os.environ.get("OUROBOROS_API_TYPE", "anthropic-messages")
        ).strip().lower()
        self._client = None         # OpenAI client (lazy)
        self._anthropic_client = None  # Anthropic client (lazy)

    def _get_client(self):
        """Return OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _get_anthropic_client(self):
        """Return Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._anthropic_client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        if self._api_type == "anthropic-messages":
            return self._chat_anthropic(messages, model, tools, max_tokens, tool_choice)
        return self._chat_openai(messages, model, tools, reasoning_effort, max_tokens, tool_choice)

    def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call via Anthropic Messages API."""
        client = self._get_anthropic_client()
        system, anth_msgs = _msgs_to_anthropic(messages)

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anth_msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            anth_tools = _tools_to_anthropic(tools)
            if anth_tools:
                kwargs["tools"] = anth_tools
                if tool_choice == "required":
                    kwargs["tool_choice"] = {"type": "any"}
                else:
                    kwargs["tool_choice"] = {"type": "auto"}

        resp = client.messages.create(**kwargs)
        return _anthropic_resp_to_openai(resp)

    def _chat_openai(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call via OpenAI-compatible API."""
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }
        # Pin Anthropic models when using OpenRouter
        if "openrouter" in self._base_url.lower() and model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if tools:
            tools_with_cache = [t for t in tools]
            if tools_with_cache:
                last_tool = {**tools_with_cache[-1]}
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        if not usage.get("cached_tokens"):
            pd = usage.get("prompt_tokens_details") or {}
            if isinstance(pd, dict) and pd.get("cached_tokens"):
                usage["cached_tokens"] = int(pd["cached_tokens"])

        if not usage.get("cache_write_tokens"):
            pd2 = usage.get("prompt_tokens_details") or {}
            if isinstance(pd2, dict):
                cw = (pd2.get("cache_write_tokens")
                      or pd2.get("cache_creation_tokens")
                      or pd2.get("cache_creation_input_tokens"))
                if cw:
                    usage["cache_write_tokens"] = int(cw)

        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anyrouter/claude-opus-4-6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anyrouter/claude-opus-4-6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anyrouter/claude-opus-4-6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
