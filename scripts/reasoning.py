#!/usr/bin/env python3
"""Chat-template inspection and rendering helpers."""

from __future__ import annotations

import hashlib
from typing import Any


def chat_template_source(tokenizer: Any) -> str:
    template = tokenizer.chat_template
    if isinstance(template, dict):
        return "\n".join(str(value) for value in template.values())
    return str(template or "")


def render_chat(tokenizer: Any, user_prompt: str, condition: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if not tokenizer.chat_template:
        return user_prompt
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if condition == "thinking":
        kwargs["enable_thinking"] = True
    elif condition == "no_thinking":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def supports_thinking_toggle(tokenizer: Any, probe_prompt: str) -> dict:
    source = chat_template_source(tokenizer)
    if "enable_thinking" not in source:
        return {
            "supports_toggle": False,
            "reason": "chat template does not reference enable_thinking",
            "rendered_on_sha256": None,
            "rendered_off_sha256": None,
        }
    try:
        rendered_on = render_chat(tokenizer, probe_prompt, "thinking")
        rendered_off = render_chat(tokenizer, probe_prompt, "no_thinking")
    except Exception as exc:
        return {
            "supports_toggle": False,
            "reason": f"template rejected enable_thinking: {exc!r}",
            "rendered_on_sha256": None,
            "rendered_off_sha256": None,
        }
    on_hash = hashlib.sha256(rendered_on.encode("utf-8")).hexdigest()
    off_hash = hashlib.sha256(rendered_off.encode("utf-8")).hexdigest()
    return {
        "supports_toggle": rendered_on != rendered_off,
        "reason": "rendered prompts differ" if rendered_on != rendered_off else "rendered prompts identical",
        "rendered_on_sha256": on_hash,
        "rendered_off_sha256": off_hash,
        "rendered_on_preview": rendered_on[:240],
        "rendered_off_preview": rendered_off[:240],
    }
