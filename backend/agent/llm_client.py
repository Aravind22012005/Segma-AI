"""Pluggable LLM client.

The agent's PLANNER and EXPLAINER stages can optionally call a real LLM for
richer natural-language understanding/generation. This module abstracts over
three providers (checked in this order) and exposes a single
`call_llm_json(...)` / `call_llm_text(...)` pair. If no API key is configured
it raises NoLLMConfigured, which callers catch to fall back to the
deterministic offline planner/explainer -- the whole app must keep working
with zero API keys set, since a hackathon demo can't depend on live quota.

Supported providers (first match wins, based on env vars):
  GEMINI_API_KEY      -> Google Gemini (gemini-flash-latest by default -- an alias Google
                          repoints at a current flash model, since pinned versions like
                          gemini-2.5-flash get retired for new API keys over time)
  OPENAI_API_KEY      -> OpenAI (gpt-4o-mini by default)
  ANTHROPIC_API_KEY   -> Anthropic Claude (claude-sonnet-5 by default)
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()


class NoLLMConfigured(Exception):
    pass


def _active_provider() -> str | None:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def llm_available() -> bool:
    return _active_provider() is not None


def active_provider_name() -> str:
    return _active_provider() or "offline (rule-based, no LLM key configured)"


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")
    return json.loads(text[start:end + 1])


def call_llm_json(system_prompt: str, user_prompt: str, model: str | None = None) -> dict:
    provider = _active_provider()
    if provider is None:
        raise NoLLMConfigured("No GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY set in environment.")

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gen_model = genai.GenerativeModel(
            model or "gemini-flash-latest",
            system_instruction=system_prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        resp = gen_model.generate_content(user_prompt)
        return _extract_json(resp.text)

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return _extract_json(resp.choices[0].message.content)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model or "claude-sonnet-5",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _extract_json(resp.content[0].text)

    raise NoLLMConfigured(f"Unknown provider '{provider}'")


def call_llm_text(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
    provider = _active_provider()
    if provider is None:
        raise NoLLMConfigured("No GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY set in environment.")

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gen_model = genai.GenerativeModel(model or "gemini-flash-latest", system_instruction=system_prompt)
        resp = gen_model.generate_content(user_prompt)
        return resp.text

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model or "claude-sonnet-5",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text

    raise NoLLMConfigured(f"Unknown provider '{provider}'")
