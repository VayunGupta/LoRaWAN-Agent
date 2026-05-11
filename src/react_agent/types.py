from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttackScenario:
    attack_type: str = "none"
    sensor: str | None = None
    gateway: str | None = None
    sensors: list[str] = field(default_factory=list)
    gateways: list[str] = field(default_factory=list)
    rssi_shift_db: float = 0.0
    noise_sigma_db: float = 0.0
    drop_prob: float = 0.0
    replay_fraction: float = 0.0
    replay_delay_s: float = 0.0
    fabricate_fraction: float = 0.0
    fabricate_shift_db: float = 8.0
    counter_shift: int = 0
    seed: int = 0

    def load_kwargs(self) -> dict[str, Any]:
        scope = "global"
        sensor_targets = self.sensors or ([self.sensor] if self.sensor else [])
        gateway_targets = self.gateways or ([self.gateway] if self.gateway else [])
        if self.attack_type == "sensor_foil":
            scope = "sensor"
        elif self.attack_type == "gateway_bias":
            scope = "gateway"
        elif self.attack_type in {
            "random_noise",
            "packet_drop",
            "selective_suppression",
            "replay_attack",
            "delayed_replay",
            "gateway_fabrication",
            "counter_corruption",
        }:
            if self.gateway:
                scope = "gateway"
            elif self.sensor:
                scope = "sensor"

        return {
            "attack_type": self.attack_type,
            "attack_scope": scope,
            "attack_sensor": self.sensor,
            "attack_gateway": self.gateway,
            "attack_sensors": sensor_targets,
            "attack_gateways": gateway_targets,
            "rssi_shift_db": self.rssi_shift_db,
            "rssi_noise_sigma_db": self.noise_sigma_db,
            "drop_prob": self.drop_prob,
            "replay_fraction": self.replay_fraction,
            "replay_delay_s": self.replay_delay_s,
            "fabricate_fraction": self.fabricate_fraction,
            "fabricate_shift_db": self.fabricate_shift_db,
            "counter_shift": self.counter_shift,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceStep:
    thought: str
    action: str
    observation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class AgentClaim:
    agent_name: str
    role: str
    label: str
    confidence: float
    rationale: str
    round_index: int = 1
    evidence_keys: list[str] = field(default_factory=list)
    request_more_evidence: bool = False
    target_sensor: str | None = None
    target_gateway: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeuristicAssessment:
    verdict: str
    predicted_attack_type: str
    confidence: float
    should_invoke_llm: bool
    trigger_reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMAdjudication:
    invoked: bool = False
    mode: str = "off"
    available: bool = False
    model: str | None = None
    endpoint: str | None = None
    final_label: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    evidence_used: list[str] = field(default_factory=list)
    request_more_evidence: bool = False
    next_tool: str | None = None
    faithful: bool = False
    fallback_reason: str | None = None
    raw_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionReport:
    architecture: str
    role_backend: str
    scenario: dict[str, Any]
    predicted_attack_type: str
    suspicious_sensor: str | None
    suspicious_gateway: str | None
    attack_detected: bool
    confidence: float
    evidence: dict[str, Any]
    heuristic: HeuristicAssessment
    llm: LLMAdjudication
    agent_claims: list[AgentClaim] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["heuristic"] = self.heuristic.to_dict()
        payload["llm"] = self.llm.to_dict()
        payload["agent_claims"] = [claim.to_dict() for claim in self.agent_claims]
        payload["trace"] = [step.to_dict() for step in self.trace]
        return payload
