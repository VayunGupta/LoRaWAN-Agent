from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from .types import LLMAdjudication


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:7b"
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    timeout_s: float = 20.0
    min_confidence: float = 0.55


class OllamaAdjudicator:
    """
    Thin bounded interface over Ollama for attack adjudication.

    The LLM sees only structured summaries derived from deterministic tools and
    must return a compact JSON object constrained to the allowed schema.
    """

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()

    def adjudicate(
        self,
        payload: dict[str, Any],
        *,
        mode: str,
        allowed_labels: list[str],
        allowed_tools: list[str],
    ) -> LLMAdjudication:
        result = LLMAdjudication(
            invoked=True,
            mode=mode,
            available=False,
            model=self.config.model,
            endpoint=self.config.endpoint,
        )
        prompt = self._build_prompt(
            payload=payload,
            mode=mode,
            allowed_labels=allowed_labels,
            allowed_tools=allowed_tools,
        )
        try:
            raw = self._generate(prompt)
        except (error.URLError, TimeoutError, ValueError, socket.timeout) as exc:
            result.fallback_reason = f"LLM unavailable: {exc}"
            return result

        result.available = True
        result.raw_response = raw
        parsed = self._parse_response(raw.get("response", ""))
        if not parsed:
            result.fallback_reason = "LLM response was not valid JSON."
            return result

        label = str(parsed.get("final_label", "")).strip()
        confidence = self._safe_float(parsed.get("confidence"), default=0.0)
        evidence_used = [str(item) for item in parsed.get("evidence_used", [])]
        next_tool = parsed.get("next_tool")
        next_tool = str(next_tool) if next_tool not in (None, "") else None
        faithful = all(item in payload["evidence"] for item in evidence_used)
        if mode == "adjudicate" and label not in allowed_labels:
            result.fallback_reason = f"LLM returned unsupported label `{label}`."
            return result
        if next_tool is not None and next_tool not in allowed_tools:
            result.fallback_reason = f"LLM requested unsupported tool `{next_tool}`."
            return result
        if confidence < self.config.min_confidence:
            result.fallback_reason = (
                f"LLM confidence {confidence:.2f} was below the cutoff {self.config.min_confidence:.2f}."
            )
            return result
        if not faithful:
            result.fallback_reason = "LLM cited evidence outside the provided payload."
            result.faithful = False
            return result

        result.final_label = label if mode == "adjudicate" else payload["heuristic"]["predicted_attack_type"]
        result.confidence = confidence
        result.rationale = str(parsed.get("rationale", "")).strip()
        result.evidence_used = evidence_used
        result.request_more_evidence = bool(parsed.get("request_more_evidence", False))
        result.next_tool = next_tool
        result.faithful = faithful
        return result

    def _generate(self, prompt: str) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "response" not in payload:
            raise ValueError("Missing `response` field from Ollama output.")
        return payload

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any] | None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_prompt(
        payload: dict[str, Any],
        *,
        mode: str,
        allowed_labels: list[str],
        allowed_tools: list[str],
    ) -> str:
        return (
            "You are a bounded LoRaWAN attack adjudicator. "
            "Use only the supplied derived evidence. Do not invent measurements, sensors, gateways, true labels, or tools.\n\n"
            f"Mode: {mode}\n"
            f"Allowed final labels: {allowed_labels}\n"
            f"Allowed next tools: {allowed_tools}\n"
            "Rules:\n"
            "- Cite evidence only by exact keys from the evidence object.\n"
            "- If evidence is conflicting, set request_more_evidence to true.\n"
            "- Prefer the heuristic label when the evidence is weak or environmentally plausible.\n"
            "- Use path context only as supporting evidence; it does not override packet-loss or geometry evidence by itself.\n"
            "- In explanation mode, keep the heuristic label and provide rationale only.\n"
            "- Return JSON only with keys: final_label, confidence, rationale, evidence_used, request_more_evidence, next_tool.\n\n"
            f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
        )
