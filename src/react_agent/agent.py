from __future__ import annotations

import json
from typing import Any

import numpy as np

from .llm import OllamaAdjudicator
from .server import GatewayCoordinatorServer
from .types import (
    AgentClaim,
    AttackScenario,
    DetectionReport,
    HeuristicAssessment,
    LLMAdjudication,
    TraceStep,
)
from .verifier import ConsistencyGraphVerifier, EnergyConsistencyVerifier


class ReActGatewayAgent:
    """
    Two-stage ReAct-style agent.

    Deterministic heuristics remain the first-stage detector. An optional LLM
    is only invoked to explain or adjudicate borderline cases where the
    heuristic evidence is ambiguous or conflicting.
    """

    ALLOWED_LABELS = [
        "none",
        "sensor_foil",
        "gateway_bias",
        "random_noise",
        "packet_drop",
        "replay_attack",
        "gateway_fabrication",
        "counter_corruption",
        "selective_suppression",
    ]
    ALLOWED_TOOLS = [
        "inspect_sensor",
        "inspect_gateway",
        "trilaterate",
        "query_gateway_agent",
        "query_temporal_agent",
        "query_physical_agent",
        "lookup_environment_context",
        "lookup_gateway_history",
        "stop",
    ]

    def __init__(
        self,
        server: GatewayCoordinatorServer,
        llm_client: OllamaAdjudicator | None = None,
        llm_mode: str = "adjudicate",
        architecture: str = "loramas",
        enabled_roles: set[str] | None = None,
        supervisor_max_rounds: int = 2,
        role_backend: str = "rules",
    ):
        self.server = server
        self.llm_client = llm_client
        self.llm_mode = llm_mode
        self.architecture = architecture
        self.role_backend = role_backend
        self.enabled_roles = enabled_roles or {
            "gateway_local",
            "temporal_consistency",
            "physical_consistency",
        }
        self.supervisor_max_rounds = max(int(supervisor_max_rounds), 1)
        self.consistency_graph_verifier = ConsistencyGraphVerifier()
        self.energy_consistency_verifier = EnergyConsistencyVerifier()

    def investigate(self, scenario: AttackScenario) -> DetectionReport:
        needs_agent_claims = self.architecture not in {"localization_only", "centralized_trust"}
        bundle = self._load_or_prepare_investigation_bundle(
            scenario=scenario,
            include_agent_claims=needs_agent_claims,
        )
        trace: list[TraceStep] = list(bundle["trace_prefix"])
        snapshot = bundle["snapshot"]
        suspicious_sensor = bundle["suspicious_sensor"]
        suspicious_gateway = bundle["suspicious_gateway"]
        sensor_view = bundle["sensor_view"]
        gateway_view = bundle["gateway_view"]
        trilateration = bundle["trilateration"]
        baseline_trilateration = bundle["baseline_trilateration"]
        evidence = dict(bundle["evidence"])
        centralized = bundle["centralized"]
        agent_claims: list[AgentClaim] = list(bundle.get("agent_claims", []))
        if self.architecture == "localization_only":
            heuristic = self._localization_only_consensus(evidence=evidence)
            trace.append(
                TraceStep(
                    thought="Use an RF/localization-only baseline that looks only at RSSI shifts, packet ratios, and trilateration residuals, without temporal, trust, or environment reasoning.",
                    action="rf_localization_baseline",
                    observation=(
                        f"Localization-only baseline predicted `{heuristic.predicted_attack_type}` with confidence "
                        f"{heuristic.confidence:.2f}. Trigger LLM: {heuristic.should_invoke_llm}. "
                        f"Reasons: {heuristic.trigger_reasons or ['none']}."
                    ),
                )
            )
        elif self.architecture == "centralized_trust":
            heuristic = self._centralized_consensus(evidence=evidence)
            trace.append(
                TraceStep(
                    thought="Use a centralized trust-consensus baseline that aggregates RF and witness features without agent-level deliberation.",
                    action="centralized_trust_consensus",
                    observation=(
                        f"Centralized baseline predicted `{heuristic.predicted_attack_type}` with confidence "
                        f"{heuristic.confidence:.2f}. Trigger LLM: {heuristic.should_invoke_llm}. "
                        f"Reasons: {heuristic.trigger_reasons or ['none']}."
                    ),
                )
            )
        else:
            if self.architecture == "consistency_graph":
                heuristic = self.consistency_graph_verifier.verify(
                    agent_claims=agent_claims,
                    evidence=evidence,
                    centralized=centralized,
                )
                trace.append(
                    TraceStep(
                        thought="The consistency-graph verifier treats attack labels as latent hypotheses and combines the three specialist claims with temporal, physical, environment-tool, and gateway-history factors.",
                        action="consistency_graph_verify",
                        observation=(
                            f"Collected {len(agent_claims)} agent claims; graph verifier predicted `{heuristic.predicted_attack_type}` "
                            f"with confidence {heuristic.confidence:.2f}. Trigger LLM: {heuristic.should_invoke_llm}. "
                            f"Reasons: {heuristic.trigger_reasons or ['none']}."
                        ),
                    )
                )
            elif self.architecture == "energy_graph":
                heuristic = self.energy_consistency_verifier.verify(
                    agent_claims=agent_claims,
                    evidence=evidence,
                    centralized=centralized,
                )
                trace.append(
                    TraceStep(
                        thought="The energy-style verifier scores each attack label with an explicit feature template over temporal, physical, gateway-history, environment, availability, and benignity evidence, then combines that with agent support.",
                        action="energy_graph_verify",
                        observation=(
                            f"Collected {len(agent_claims)} agent claims; energy verifier predicted `{heuristic.predicted_attack_type}` "
                            f"with confidence {heuristic.confidence:.2f}. Trigger LLM: {heuristic.should_invoke_llm}. "
                            f"Reasons: {heuristic.trigger_reasons or ['none']}."
                        ),
                    )
                )
            else:
                heuristic = self._supervisor_consensus(
                    agent_claims=agent_claims,
                    evidence=evidence,
                )
                trace.append(
                    TraceStep(
                        thought="LoRaMAS collects independent claims from gateway, temporal, and physical agents, while environment context and gateway history remain tool evidence for the supervisor.",
                        action="loramas_supervisor_round1",
                        observation=(
                            f"Collected {len(agent_claims)} agent claims; supervisor predicted `{heuristic.predicted_attack_type}` "
                            f"with confidence {heuristic.confidence:.2f}. Trigger LLM: {heuristic.should_invoke_llm}. "
                            f"Reasons: {heuristic.trigger_reasons or ['none']}."
                        ),
                    )
                )

        llm = self._default_llm_result()
        final_label = heuristic.predicted_attack_type
        final_confidence = heuristic.confidence

        if self.llm_mode != "off" and heuristic.should_invoke_llm:
            llm_payload = self._build_llm_payload(
                scenario=scenario,
                heuristic=heuristic,
                suspicious_sensor=suspicious_sensor,
                suspicious_gateway=suspicious_gateway,
                sensor_view=sensor_view,
                gateway_view=gateway_view,
                trilateration=trilateration,
                baseline_trilateration=baseline_trilateration,
                agent_claims=agent_claims,
            )
            llm = self._run_llm(llm_payload)
            if llm.available and llm.fallback_reason is None:
                trace.append(
                    TraceStep(
                        thought="The LoRaMAS supervisor found unresolved conflicts, so escalate to the LLM only as a bounded debate arbiter over structured agent claims and tool outputs.",
                        action=f"llm_{self.llm_mode}",
                        observation=(
                            f"LLM returned label `{llm.final_label}` with confidence {llm.confidence:.2f}, "
                            f"requested more evidence={llm.request_more_evidence}, and cited {llm.evidence_used}."
                        ),
                    )
                )
                if self.llm_mode == "adjudicate" and llm.final_label is not None:
                    final_label = llm.final_label
                    final_confidence = float(llm.confidence or heuristic.confidence)
            else:
                trace.append(
                    TraceStep(
                        thought="The LLM is optional, so malformed or unavailable responses must fall back to the deterministic detector.",
                        action=f"llm_{self.llm_mode}_fallback",
                        observation=llm.fallback_reason or "LLM was skipped.",
                    )
                )

        attack_detected = final_label != "none" or llm.request_more_evidence
        return DetectionReport(
            architecture=self.architecture,
            role_backend=self.role_backend,
            scenario=scenario.to_dict(),
            predicted_attack_type=final_label,
            suspicious_sensor=suspicious_sensor,
            suspicious_gateway=suspicious_gateway,
            attack_detected=attack_detected,
            confidence=final_confidence,
            evidence=heuristic.evidence,
            heuristic=heuristic,
            llm=llm,
            agent_claims=agent_claims,
            trace=trace,
        )

    def _load_or_prepare_investigation_bundle(
        self,
        *,
        scenario: AttackScenario,
        include_agent_claims: bool,
    ) -> dict[str, Any]:
        cache_key = json.dumps(
            {
                "scenario": scenario.to_dict(),
                "enabled_roles": sorted(self.enabled_roles),
                "include_agent_claims": bool(include_agent_claims),
                "role_backend": self.role_backend if include_agent_claims else "none",
            },
            sort_keys=True,
        )
        cached = self.server._investigation_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        trace_prefix: list[TraceStep] = []
        snapshot = self.server.get_network_snapshot(scenario)
        delta = snapshot["delta"]
        trust = snapshot["trust"]
        sensor_rank = self.server.rank_sensors(delta)
        gateway_rank = self.server.rank_gateways(delta)

        trace_prefix.append(
            TraceStep(
                thought="Start from a network-wide snapshot so I can see whether anomalies cluster by sensor, gateway, or packet volume.",
                action="load_network_snapshot",
                observation=(
                    f"Compared {int(delta.shape[0])} sensor-gateway links against the clean baseline; "
                    f"top sensor anomaly score is {self._top_score(sensor_rank):.2f} and top gateway anomaly score is {self._top_score(gateway_rank):.2f}."
                ),
            )
        )

        suspicious_sensor = self._choose_sensor(sensor_rank, scenario)
        suspicious_gateway = self._choose_gateway(gateway_rank, scenario)
        if trust.get("suspicious_sensor") and trust.get("max_replay_gap", 0.0) >= 1.0:
            suspicious_sensor = trust["suspicious_sensor"]
        if trust.get("suspicious_gateway") and (
            trust.get("fabricated_witness_count", 0) > 0
            or trust.get("event_inconsistency_score", 0.0) >= 1.5
        ):
            suspicious_gateway = trust["suspicious_gateway"]

        sensor_view = self.server.sensor_view(delta, suspicious_sensor) if suspicious_sensor else None
        gateway_view = self.server.gateway_view(delta, suspicious_gateway) if suspicious_gateway else None

        if sensor_view is not None and not sensor_view.empty:
            trace_prefix.append(
                TraceStep(
                    thought="Inspect the most suspicious sensor across all gateways to see whether the evidence is consistent with a compromised end device.",
                    action=f"inspect_sensor({suspicious_sensor})",
                    observation=self._summarize_sensor_view(sensor_view),
                )
            )

        if gateway_view is not None and not gateway_view.empty:
            trace_prefix.append(
                TraceStep(
                    thought="Check whether one gateway is producing the same anomaly for many different sensors, which would point to gateway-side corruption.",
                    action=f"inspect_gateway({suspicious_gateway})",
                    observation=self._summarize_gateway_view(gateway_view),
                )
            )

        trilateration = (
            self.server.trilateration_view(snapshot["packets"], suspicious_sensor)
            if suspicious_sensor
            else {"available": False, "reason": "No suspicious sensor was selected."}
        )
        baseline_trilateration = (
            self.server.baseline_trilateration_view(suspicious_sensor)
            if suspicious_sensor
            else {"available": False, "reason": "No suspicious sensor was selected."}
        )

        trace_prefix.append(
            TraceStep(
                thought="Use trilateration as a physics-grounded cross-check; a real attack should distort the inferred geometry, not just one scalar feature.",
                action=f"trilaterate({suspicious_sensor})" if suspicious_sensor else "trilaterate(None)",
                observation=self._summarize_trilateration(trilateration, baseline_trilateration),
            )
        )

        trace_prefix.append(
            TraceStep(
                thought="Compare packet witnesses across gateways so replayed, fabricated, or suppressed observations can be detected from cross-gateway disagreement rather than RSSI alone.",
                action="analyze_packet_witness_consistency",
                observation=self._summarize_trust(trust),
            )
        )

        evidence = self._build_evidence(
            sensor_view=sensor_view,
            gateway_view=gateway_view,
            sensor_rank=sensor_rank,
            gateway_rank=gateway_rank,
            trilateration=trilateration,
            baseline_trilateration=baseline_trilateration,
            trust=trust,
        )
        if any(
            key in evidence
            for key in [
                "sensor_context_fragility",
                "gateway_context_fragility",
                "sensor_environment_plausibility",
                "gateway_environment_plausibility",
            ]
        ):
            trace_prefix.append(
                TraceStep(
                    thought="Check whether satellite-derived obstruction and fragility make the RF mismatch physically plausible before accusing a sensor or gateway.",
                    action="inspect_environment_context",
                    observation=self._summarize_environment_evidence(evidence),
                )
            )

        agent_claims: list[AgentClaim] = []
        if include_agent_claims:
            agent_claims = self._run_loramas_agents(
                snapshot=snapshot,
                scenario=scenario,
                suspicious_sensor=suspicious_sensor,
                suspicious_gateway=suspicious_gateway,
                sensor_view=sensor_view,
                gateway_view=gateway_view,
                trilateration=trilateration,
                baseline_trilateration=baseline_trilateration,
                evidence=evidence,
            )

        bundle = {
            "trace_prefix": trace_prefix,
            "snapshot": snapshot,
            "suspicious_sensor": suspicious_sensor,
            "suspicious_gateway": suspicious_gateway,
            "sensor_view": sensor_view,
            "gateway_view": gateway_view,
            "trilateration": trilateration,
            "baseline_trilateration": baseline_trilateration,
            "evidence": dict(evidence),
            "agent_claims": agent_claims,
            "centralized": self._centralized_consensus(evidence=evidence),
        }
        self.server._investigation_bundle_cache[cache_key] = bundle
        return bundle

    def _run_llm(self, payload: dict[str, Any]) -> LLMAdjudication:
        if self.llm_client is None:
            return LLMAdjudication(
                invoked=True,
                mode=self.llm_mode,
                available=False,
                fallback_reason="No LLM client was configured.",
            )
        return self.llm_client.adjudicate(
            payload,
            mode=self.llm_mode,
            allowed_labels=self.ALLOWED_LABELS,
            allowed_tools=self.ALLOWED_TOOLS,
        )

    def _default_llm_result(self) -> LLMAdjudication:
        return LLMAdjudication(invoked=False, mode=self.llm_mode)

    def _build_llm_payload(
        self,
        *,
        scenario: AttackScenario,
        heuristic: HeuristicAssessment,
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
        sensor_view,
        gateway_view,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
        agent_claims: list[AgentClaim],
    ) -> dict[str, Any]:
        del scenario
        evidence = heuristic.evidence
        return {
            "case_context": {
                "suspicious_sensor": suspicious_sensor,
                "suspicious_gateway": suspicious_gateway,
                "heuristic_prediction": heuristic.predicted_attack_type,
                "heuristic_verdict": heuristic.verdict,
                "heuristic_confidence": round(float(heuristic.confidence), 3),
                "trigger_reasons": heuristic.trigger_reasons,
            },
            "evidence": self._select_llm_evidence(evidence),
            "agent_claims": [claim.to_dict() for claim in agent_claims],
            "sensor_path_context": self._compact_path_context(sensor_view),
            "gateway_path_context": self._compact_path_context(gateway_view),
            "trilateration_delta": self._compact_trilateration_delta(
                trilateration=trilateration,
                baseline_trilateration=baseline_trilateration,
            ),
        }

    def _run_loramas_agents(
        self,
        *,
        snapshot: dict[str, Any],
        scenario: AttackScenario,
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
        sensor_view,
        gateway_view,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
        evidence: dict[str, Any],
    ) -> list[AgentClaim]:
        claims = self._initial_agent_claims(
            snapshot=snapshot,
            scenario=scenario,
            suspicious_sensor=suspicious_sensor,
            suspicious_gateway=suspicious_gateway,
            evidence=evidence,
            trilateration=trilateration,
            baseline_trilateration=baseline_trilateration,
        )
        if self.role_backend == "llm":
            claims = [
                self._llm_role_claim(claim=claim, evidence=evidence)
                for claim in claims
            ]
        if self.supervisor_max_rounds <= 1:
            return claims

        for round_index in range(2, self.supervisor_max_rounds + 1):
            rebuttals = self._agent_rebuttal_round(
                claims=claims,
                evidence=evidence,
                suspicious_sensor=suspicious_sensor,
                suspicious_gateway=suspicious_gateway,
                round_index=round_index,
            )
            if not rebuttals:
                break
            claims.extend(rebuttals)
        return claims

    def _llm_role_claim(
        self,
        *,
        claim: AgentClaim,
        evidence: dict[str, Any],
    ) -> AgentClaim:
        if self.llm_client is None:
            claim.metadata = {
                **claim.metadata,
                "role_backend": "rules",
                "llm_role_fallback": "No LLM client was configured.",
            }
            return claim
        payload = {
            "case_context": {
                "role": claim.role,
                "agent_name": claim.agent_name,
                "rule_label": claim.label,
                "rule_confidence": round(float(claim.confidence), 3),
                "rule_rationale": claim.rationale,
                "target_sensor": claim.target_sensor,
                "target_gateway": claim.target_gateway,
            },
            "evidence": self._select_llm_evidence(evidence),
            "agent_claims": [claim.to_dict()],
        }
        llm = self.llm_client.adjudicate(
            payload,
            mode="adjudicate",
            allowed_labels=self.ALLOWED_LABELS,
            allowed_tools=self.ALLOWED_TOOLS,
        )
        if not llm.available or llm.fallback_reason is not None or llm.final_label is None:
            claim.metadata = {
                **claim.metadata,
                "role_backend": "rules",
                "llm_role_fallback": llm.fallback_reason or "LLM role claim unavailable.",
            }
            return claim
        return AgentClaim(
            agent_name=f"{claim.agent_name}_llm",
            role=claim.role,
            label=llm.final_label,
            confidence=float(llm.confidence or claim.confidence),
            rationale=llm.rationale or claim.rationale,
            round_index=claim.round_index,
            evidence_keys=llm.evidence_used or claim.evidence_keys,
            request_more_evidence=bool(llm.request_more_evidence),
            target_sensor=claim.target_sensor,
            target_gateway=claim.target_gateway,
            metadata={
                **claim.metadata,
                "role_backend": "llm",
                "rule_label": claim.label,
                "rule_confidence": round(float(claim.confidence), 3),
                "llm_role_model": llm.model,
            },
        )

    def _initial_agent_claims(
        self,
        *,
        snapshot: dict[str, Any],
        scenario: AttackScenario,
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
        evidence: dict[str, Any],
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
    ) -> list[AgentClaim]:
        claims: list[AgentClaim] = []
        gateways = sorted(self.server.metadata["GATEWAYS_LATLON"].keys())
        delta = snapshot["delta"]
        trust = snapshot["trust"]
        if "gateway_local" in self.enabled_roles:
            for gateway in gateways:
                local_view = self.server.gateway_view(delta, gateway)
                claims.append(
                    self._gateway_local_agent_claim(
                        gateway=gateway,
                        gateway_view=local_view,
                        trust=trust,
                        suspicious_sensor=suspicious_sensor,
                    )
                )
        if "temporal_consistency" in self.enabled_roles:
            claims.append(
                self._temporal_agent_claim(
                    trust=trust,
                    suspicious_sensor=suspicious_sensor,
                    suspicious_gateway=suspicious_gateway,
                )
            )
        if "physical_consistency" in self.enabled_roles:
            claims.append(
                self._physical_agent_claim(
                    evidence=evidence,
                    suspicious_sensor=suspicious_sensor,
                    suspicious_gateway=suspicious_gateway,
                    trilateration=trilateration,
                    baseline_trilateration=baseline_trilateration,
                )
            )
        return claims

    def _agent_rebuttal_round(
        self,
        *,
        claims: list[AgentClaim],
        evidence: dict[str, Any],
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
        round_index: int,
    ) -> list[AgentClaim]:
        base_claims = [claim for claim in claims if claim.round_index == 1]
        labels = {claim.label for claim in base_claims if claim.label != "none"}
        if len(labels) <= 1:
            return []
        ranked = self._rank_labels(base_claims)
        if len(ranked) < 2:
            return []
        top_label, top_score = ranked[0]
        runner_up, runner_score = ranked[1]
        if top_score - runner_score >= 0.45:
            return []

        rebuttals: list[AgentClaim] = []
        for claim in base_claims:
            if claim.role == "gateway_local":
                gateway_trust = float(claim.metadata.get("gateway_trust", 1.0))
                if claim.target_gateway == suspicious_gateway and top_label in {
                    "gateway_fabrication",
                    "counter_corruption",
                    "selective_suppression",
                }:
                    rebuttal_label = claim.label if claim.label != "none" else top_label
                    rebuttal_confidence = claim.confidence
                    rebuttal_rationale = (
                        f"Local gateway agent for {claim.target_gateway} challenges the supervisor focus on `{top_label}` "
                        "and argues that its local packet stream remains more self-consistent than the competing accusation suggests."
                    )
                    if gateway_trust >= 0.92:
                        rebuttal_label = "none"
                        rebuttal_confidence = self._bounded_confidence(min(claim.confidence + 0.07, 0.88))
                        rebuttal_rationale = (
                            f"Local gateway agent for {claim.target_gateway} disputes the `{top_label}` accusation because its trust score remains relatively high and its local view lacks a decisive gateway-side failure signature."
                        )
                    rebuttals.append(
                        AgentClaim(
                            agent_name=f"gateway_agent_{claim.target_gateway}_rebuttal",
                            role="gateway_local",
                            label=rebuttal_label,
                            confidence=rebuttal_confidence,
                            rationale=rebuttal_rationale,
                            round_index=round_index,
                            evidence_keys=["min_gateway_trust", "gateway_worst_packet_ratio", "gateway_mean_abs_rssi_shift_db"],
                            request_more_evidence=False,
                            target_sensor=suspicious_sensor,
                            target_gateway=claim.target_gateway,
                            metadata={"rebuttal_against": top_label, "gateway_trust": gateway_trust},
                        )
                    )
            elif claim.role == "temporal_consistency" and claim.label not in {top_label, "none"}:
                rebuttals.append(
                    AgentClaim(
                        agent_name="temporal_agent_rebuttal",
                        role="temporal_consistency",
                        label=claim.label,
                        confidence=self._bounded_confidence(min(claim.confidence + 0.05, 0.92)),
                        rationale=(
                            f"Temporal evidence still favors `{claim.label}` over `{top_label}` because replay and counter-order signals are less explainable by purely RF hypotheses."
                        ),
                        round_index=round_index,
                        evidence_keys=["max_replay_gap", "replay_event_count", "counter_regression_count"],
                        request_more_evidence=False,
                        target_sensor=suspicious_sensor,
                        target_gateway=suspicious_gateway,
                        metadata={"rebuttal_against": top_label},
                    )
                )
            elif claim.role == "physical_consistency" and claim.label not in {top_label, "none"}:
                rebuttals.append(
                    AgentClaim(
                        agent_name="physical_agent_rebuttal",
                        role="physical_consistency",
                        label=claim.label,
                        confidence=self._bounded_confidence(min(claim.confidence + 0.05, 0.92)),
                        rationale=(
                            f"Physical evidence still favors `{claim.label}` over `{top_label}` because RF shifts and geometry changes remain stronger than the competing explanation."
                        ),
                        round_index=round_index,
                        evidence_keys=[
                            "sensor_mean_abs_rssi_shift_db",
                            "gateway_mean_abs_rssi_shift_db",
                            "trilateration_residual_delta_m",
                        ],
                        request_more_evidence=False,
                        target_sensor=suspicious_sensor,
                        target_gateway=suspicious_gateway,
                        metadata={"rebuttal_against": top_label},
                    )
                )
        return rebuttals

    def _gateway_local_agent_claim(
        self,
        *,
        gateway: str,
        gateway_view,
        trust: dict[str, Any],
        suspicious_sensor: str | None,
    ) -> AgentClaim:
        mean_abs_shift = self._safe_mean_abs(gateway_view, "rssi_shift_db")
        min_ratio = self._safe_min(gateway_view, "packet_ratio", default=1.0)
        ratio_std = self._safe_std(gateway_view, "packet_ratio")
        low_ratio_sensor_count = (
            int((gateway_view["packet_ratio"] < 0.7).sum())
            if gateway_view is not None and not gateway_view.empty
            else 0
        )
        gateway_sensor_count = int(gateway_view.shape[0]) if gateway_view is not None and not gateway_view.empty else 0
        impacted = (
            int((gateway_view["rssi_shift_db"].abs() >= 6.0).sum())
            if gateway_view is not None and not gateway_view.empty
            else 0
        )
        gateway_trust = float(trust.get("gateway_trust_scores", {}).get(gateway, 1.0))
        fabricated = int(trust.get("fabricated_witness_count", 0))
        multiplicity = int(trust.get("multiplicity_anomaly_count", 0))
        event_inconsistency = float(trust.get("event_inconsistency_score", 0.0))
        label = "none"
        confidence = 0.62
        rationale = "Local gateway view appears consistent with benign behavior."
        evidence_keys = ["min_gateway_trust", "gateway_mean_abs_rssi_shift_db"]
        if fabricated >= 5 and gateway == trust.get("suspicious_gateway") and gateway_trust < 0.9:
            label = "gateway_fabrication"
            confidence = self._bounded_confidence(0.72 + min((1.0 - gateway_trust), 0.18))
            rationale = "This gateway is the least trusted observer and is aligned with fabrication-like witness inconsistencies."
            evidence_keys = ["fabricated_witness_count", "min_gateway_trust"]
        elif (
            multiplicity >= 3
            and min_ratio >= 0.7
            and gateway == trust.get("suspicious_gateway")
            and gateway_trust < 0.94
            and event_inconsistency >= 0.8
        ):
            label = "counter_corruption"
            confidence = self._bounded_confidence(0.68 + min((1.0 - gateway_trust), 0.16))
            rationale = "This gateway disagrees with the others primarily through witness identity and multiplicity rather than pure packet loss."
            evidence_keys = ["multiplicity_anomaly_count", "min_gateway_trust", "event_inconsistency_score"]
        elif (
            min_ratio < 0.55
            and low_ratio_sensor_count <= max(1, gateway_sensor_count // 3)
            and ratio_std >= 0.12
        ):
            label = "selective_suppression"
            confidence = self._bounded_confidence(0.74 + (1.0 - min_ratio) * 0.22)
            rationale = "Loss is concentrated on a subset of this gateway's sensor links rather than spread uniformly, which looks more selective than a broad drop."
            evidence_keys = ["gateway_worst_packet_ratio", "gateway_packet_ratio_std", "gateway_low_ratio_sensor_count"]
        elif min_ratio < 0.55:
            label = "packet_drop"
            confidence = self._bounded_confidence(0.73 + (1.0 - min_ratio) * 0.2)
            rationale = "The gateway's packet-retention pattern is broadly depressed across links, which looks more like generalized drop than selective suppression."
            evidence_keys = ["gateway_worst_packet_ratio", "gateway_packet_ratio_std", "gateway_low_ratio_sensor_count"]
        elif mean_abs_shift >= 6.0 and impacted >= 4:
            label = "gateway_bias"
            confidence = self._bounded_confidence(0.74 + min(mean_abs_shift / 30.0, 0.18))
            rationale = "The gateway reports a coherent RF shift across multiple sensors, which is consistent with gateway-side bias."
            evidence_keys = ["gateway_mean_abs_rssi_shift_db", "gateway_impacted_sensor_count"]
        return AgentClaim(
            agent_name=f"gateway_agent_{gateway}",
            role="gateway_local",
            label=label,
            confidence=confidence,
            rationale=rationale,
            round_index=1,
            evidence_keys=evidence_keys,
            request_more_evidence=label == "none" and gateway == trust.get("suspicious_gateway"),
            target_sensor=suspicious_sensor,
            target_gateway=gateway,
            metadata={"gateway_trust": round(gateway_trust, 3)},
        )

    def _temporal_agent_claim(
        self,
        *,
        trust: dict[str, Any],
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
    ) -> AgentClaim:
        max_replay_gap = float(trust.get("max_replay_gap", 0.0))
        replay_events = int(trust.get("replay_event_count", 0))
        regressions = int(trust.get("counter_regression_count", 0))
        delayed_duplicates = int(trust.get("delayed_duplicate_count", 0))
        max_duplicate_gap_s = float(trust.get("max_duplicate_gap_s", 0.0))
        label = "none"
        confidence = 0.64
        rationale = "Temporal packet ordering does not indicate a strong replay signal."
        evidence_keys = ["max_replay_gap", "replay_event_count"]
        if max_replay_gap >= 3.0 or replay_events >= 2:
            label = "replay_attack"
            confidence = self._bounded_confidence(0.78 + min(max_replay_gap / 10.0, 0.18))
            rationale = "Recent packet witnesses reintroduced old counters after newer ones were already seen, which is consistent with replay."
        elif max_replay_gap >= 1.0 and regressions >= 10:
            label = "replay_attack"
            confidence = self._bounded_confidence(0.74 + min(regressions / 200.0, 0.16))
            rationale = "A large number of timestamp-local counter regressions indicates repeated reintroduction of older packets, which is more consistent with replay than benign drift."
            evidence_keys = ["counter_regression_count", "max_replay_gap"]
        elif delayed_duplicates >= 20 and max_duplicate_gap_s >= 30.0:
            label = "replay_attack"
            confidence = self._bounded_confidence(0.72 + min(max_duplicate_gap_s / 300.0, 0.16))
            rationale = "The same sensor-counter witness is reappearing at one gateway after a substantial delay, which is a direct replay pattern even without a strong global counter regression."
            evidence_keys = ["delayed_duplicate_count", "max_duplicate_gap_s"]
        elif regressions >= 3:
            label = "counter_corruption"
            confidence = self._bounded_confidence(0.70 + min(regressions / 20.0, 0.16))
            rationale = "The counter stream shows repeated identity discontinuities without a dominant RF explanation."
            evidence_keys = ["counter_regression_count", "max_replay_gap"]
        return AgentClaim(
            agent_name="temporal_agent",
            role="temporal_consistency",
            label=label,
            confidence=confidence,
            rationale=rationale,
            round_index=1,
            evidence_keys=evidence_keys,
            request_more_evidence=label == "none" and suspicious_sensor is not None,
            target_sensor=suspicious_sensor,
            target_gateway=suspicious_gateway,
        )

    def _physical_agent_claim(
        self,
        *,
        evidence: dict[str, Any],
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
    ) -> AgentClaim:
        sensor_shift = float(evidence["sensor_mean_abs_rssi_shift_db"])
        gateway_shift = float(evidence["gateway_mean_abs_rssi_shift_db"])
        sensor_ratio = float(evidence["sensor_worst_packet_ratio"])
        gateway_ratio = float(evidence["gateway_worst_packet_ratio"])
        residual_delta = float(evidence["trilateration_residual_delta_m"])
        std_delta = float(evidence["sensor_mean_std_delta_db"])
        gateway_impacted = int(evidence["gateway_impacted_sensor_count"])
        top_sensor = float(evidence.get("top_sensor_score", 0.0))
        top_gateway = float(evidence.get("top_gateway_score", 0.0))
        gateway_packet_ratio_std = float(evidence.get("gateway_packet_ratio_std", 0.0))
        gateway_low_ratio_sensor_count = int(evidence.get("gateway_low_ratio_sensor_count", 0))
        gateway_sensor_count = int(evidence.get("gateway_sensor_count", 0))
        label = "none"
        confidence = 0.65
        rationale = "RF and geometry evidence remain within the benign envelope."
        evidence_keys = ["top_sensor_score", "top_gateway_score"]
        if gateway_shift >= 6.0 and gateway_impacted >= 4 and gateway_shift > sensor_shift:
            label = "gateway_bias"
            confidence = self._bounded_confidence(0.77 + min(gateway_shift / 25.0, 0.16))
            rationale = "Physical evidence shows coherent gateway-wide shift across multiple sensors."
            evidence_keys = ["gateway_mean_abs_rssi_shift_db", "gateway_impacted_sensor_count"]
        elif gateway_shift >= 3.5 and top_gateway >= 5.0 and gateway_shift > (sensor_shift + 0.75):
            label = "gateway_bias"
            confidence = self._bounded_confidence(0.72 + min(gateway_shift / 20.0, 0.14))
            rationale = "Even though the shift is weaker, the evidence still concentrates coherently at one gateway rather than appearing as diffuse noise."
            evidence_keys = ["gateway_mean_abs_rssi_shift_db", "top_gateway_score"]
        elif sensor_shift >= 6.0 and sensor_shift > gateway_shift:
            label = "sensor_foil"
            confidence = self._bounded_confidence(0.76 + min(sensor_shift / 25.0, 0.16))
            rationale = "Physical evidence is concentrated on one sensor across gateways, which is more consistent with a sensor-side anomaly."
            evidence_keys = ["sensor_mean_abs_rssi_shift_db", "top_sensor_score"]
        elif sensor_shift >= 3.5 and top_sensor >= 5.0 and sensor_shift > (gateway_shift + 0.75):
            label = "sensor_foil"
            confidence = self._bounded_confidence(0.72 + min(sensor_shift / 20.0, 0.14))
            rationale = "The weaker RF shift is still directional and sensor-centered, which fits a sensor-side disturbance better than diffuse random noise."
            evidence_keys = ["sensor_mean_abs_rssi_shift_db", "top_sensor_score"]
        elif (
            sensor_ratio < 0.55 or gateway_ratio < 0.55
        ) and gateway_low_ratio_sensor_count <= max(1, gateway_sensor_count // 3) and gateway_packet_ratio_std >= 0.12:
            label = "selective_suppression"
            confidence = self._bounded_confidence(0.74 + (1.0 - min(sensor_ratio, gateway_ratio)) * 0.18)
            rationale = "Loss is concentrated on only a subset of the gateway's links, which fits selective suppression better than broad packet drop."
            evidence_keys = ["sensor_worst_packet_ratio", "gateway_worst_packet_ratio", "gateway_packet_ratio_std"]
        elif sensor_ratio < 0.55 or gateway_ratio < 0.55:
            label = "packet_drop"
            confidence = self._bounded_confidence(0.74 + (1.0 - min(sensor_ratio, gateway_ratio)) * 0.18)
            rationale = "Packet-retention ratios indicate strong loss on the affected links."
            evidence_keys = ["sensor_worst_packet_ratio", "gateway_worst_packet_ratio"]
        elif (std_delta >= 2.0 and max(sensor_shift, gateway_shift) < 2.5) or (
            residual_delta >= 10.0 and max(top_sensor, top_gateway) < 5.0
        ):
            label = "random_noise"
            confidence = self._bounded_confidence(0.72 + min(max(std_delta, residual_delta / 10.0) / 10.0, 0.16))
            rationale = "The dominant signal is variance inflation or geometry degradation rather than a coherent directional shift."
            evidence_keys = ["sensor_mean_std_delta_db", "trilateration_residual_delta_m"]
        elif not trilateration.get("available") or not baseline_trilateration.get("available"):
            confidence = 0.58
            rationale = "Physical reasoning is limited because trilateration is unavailable for this focus case."
        return AgentClaim(
            agent_name="physical_agent",
            role="physical_consistency",
            label=label,
            confidence=confidence,
            rationale=rationale,
            round_index=1,
            evidence_keys=evidence_keys,
            request_more_evidence=label == "none",
            target_sensor=suspicious_sensor,
            target_gateway=suspicious_gateway,
        )

    def _supervisor_consensus(
        self,
        *,
        agent_claims: list[AgentClaim],
        evidence: dict[str, Any],
    ) -> HeuristicAssessment:
        role_weights = {
            "gateway_local": 1.0,
            "temporal_consistency": 1.35,
            "physical_consistency": 1.35,
        }
        label_scores, label_support = self._rank_labels(
            agent_claims,
            role_weights=role_weights,
            with_support=True,
            none_role_discounts={
                "gateway_local": 0.2,
                "physical_consistency": 0.75,
                "temporal_consistency": 0.75,
            },
        )
        ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_attack_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        max_replay_gap = float(evidence.get("max_replay_gap", 0.0))
        max_duplicate_gap_s = float(evidence.get("max_duplicate_gap_s", 0.0))
        delayed_duplicate_count = int(evidence.get("delayed_duplicate_count", 0))
        counter_regressions = int(evidence.get("counter_regression_count", 0))
        replay_support = float(label_scores.get("replay_attack", 0.0))
        centralized = self._centralized_consensus(evidence=evidence)
        if (
            predicted_attack_type == "none"
            and max_replay_gap >= 1.0
            and counter_regressions >= 10
            and replay_support >= 1.0
        ):
            predicted_attack_type = "replay_attack"
            top_score = replay_support
            second_score = max(
                score for label, score in label_scores.items() if label != predicted_attack_type
            )
        elif (
            predicted_attack_type == "none"
            and delayed_duplicate_count >= 20
            and max_duplicate_gap_s >= 30.0
            and replay_support >= 0.8
        ):
            predicted_attack_type = "replay_attack"
            top_score = max(replay_support, 1.1)
            second_score = max(
                score for label, score in label_scores.items() if label != predicted_attack_type
            )
        elif (
            predicted_attack_type == "none"
            and centralized.predicted_attack_type != "none"
            and centralized.confidence >= 0.78
        ):
            predicted_attack_type = centralized.predicted_attack_type
            top_score = max(top_score, centralized.confidence * 1.5)
            second_score = max(
                score for label, score in label_scores.items() if label != predicted_attack_type
            )
        margin = top_score - second_score
        confidence = self._bounded_confidence(0.55 + min(top_score / 4.0, 0.24) + min(max(margin, 0.0) / 3.0, 0.16))
        verdict = "accept" if predicted_attack_type == "none" else "reject"
        reasons: list[str] = []
        if margin < 0.35:
            reasons.append("inter_agent_conflict")
        if len(label_support.get(predicted_attack_type, [])) < 2 and predicted_attack_type != "none":
            reasons.append("weak_multiagent_support")
        if predicted_attack_type == "none" and float(evidence.get("top_sensor_score", 0.0)) >= 2.5:
            reasons.append("rf_layer_still_suspicious")
        if predicted_attack_type == "none" and float(evidence.get("event_inconsistency_score", 0.0)) >= 1.2:
            reasons.append("gateway_history_still_suspicious")
        if float(evidence.get("max_replay_gap", 0.0)) >= 2.0 and predicted_attack_type != "replay_attack":
            reasons.append("temporal_agent_disagrees")
        if delayed_duplicate_count >= 20 and max_duplicate_gap_s >= 30.0 and predicted_attack_type != "replay_attack":
            reasons.append("duplicate_replay_signal")
        if predicted_attack_type != "none" and max(
            float(evidence.get("sensor_environment_plausibility", 0.0)),
            float(evidence.get("gateway_environment_plausibility", 0.0)),
        ) >= 0.7:
            reasons.append("environment_context_disagrees")
        if float(evidence.get("min_gateway_trust", 1.0)) < 0.85 and predicted_attack_type == "none":
            reasons.append("gateway_agent_disagreement")
        should_invoke_llm = bool(reasons)
        if should_invoke_llm:
            verdict = "uncertain"
        enriched_evidence = dict(evidence)
        enriched_evidence["loramas_vote_margin"] = round(float(margin), 3)
        enriched_evidence["loramas_top_label_score"] = round(float(top_score), 3)
        enriched_evidence["loramas_runner_up_score"] = round(float(second_score), 3)
        enriched_evidence["loramas_supporting_agents"] = label_support.get(predicted_attack_type, [])
        return HeuristicAssessment(
            verdict=verdict,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            should_invoke_llm=should_invoke_llm,
            trigger_reasons=reasons,
            evidence=enriched_evidence,
        )

    @staticmethod
    def _rank_labels(
        agent_claims: list[AgentClaim],
        role_weights: dict[str, float] | None = None,
        with_support: bool = False,
        none_role_discounts: dict[str, float] | None = None,
    ):
        role_weights = role_weights or {}
        none_role_discounts = none_role_discounts or {}
        label_scores: dict[str, float] = {}
        label_support: dict[str, list[str]] = {}
        for claim in agent_claims:
            weight = role_weights.get(claim.role, 1.0) * max(float(claim.confidence), 0.0)
            if claim.label == "none":
                weight *= none_role_discounts.get(claim.role, 1.0)
            label_scores[claim.label] = label_scores.get(claim.label, 0.0) + weight
            label_support.setdefault(claim.label, []).append(claim.agent_name)
        ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        if with_support:
            return label_scores, label_support
        return ranked

    def _centralized_consensus(
        self,
        *,
        evidence: dict[str, Any],
    ) -> HeuristicAssessment:
        predicted_attack_type = "none"
        confidence = 0.6
        verdict = "uncertain"
        top_sensor = float(evidence["top_sensor_score"])
        top_gateway = float(evidence["top_gateway_score"])
        top_sensor_raw = float(evidence.get("top_sensor_score_raw", top_sensor))
        top_gateway_raw = float(evidence.get("top_gateway_score_raw", top_gateway))
        max_plausibility = max(
            float(evidence.get("sensor_environment_plausibility", 0.0)),
            float(evidence.get("gateway_environment_plausibility", 0.0)),
        )
        max_replay_gap = float(evidence.get("max_replay_gap", 0.0))
        replay_event_count = int(evidence.get("replay_event_count", 0))
        counter_regressions = int(evidence.get("counter_regression_count", 0))
        delayed_duplicate_count = int(evidence.get("delayed_duplicate_count", 0))
        max_duplicate_gap_s = float(evidence.get("max_duplicate_gap_s", 0.0))
        fabricated = int(evidence.get("fabricated_witness_count", 0))
        multiplicity = int(evidence.get("multiplicity_anomaly_count", 0))
        min_gateway_trust = float(evidence.get("min_gateway_trust", 1.0))
        sensor_ratio = float(evidence["sensor_worst_packet_ratio"])
        gateway_ratio = float(evidence["gateway_worst_packet_ratio"])
        sensor_shift = float(evidence["sensor_mean_abs_rssi_shift_db"])
        gateway_shift = float(evidence["gateway_mean_abs_rssi_shift_db"])
        gateway_impacted = int(evidence["gateway_impacted_sensor_count"])
        std_delta = float(evidence["sensor_mean_std_delta_db"])
        residual_delta = float(evidence["trilateration_residual_delta_m"])
        event_score = float(evidence.get("event_inconsistency_score", 0.0))

        if max(top_sensor, top_gateway) < 2.2 and max_plausibility >= 0.65:
            predicted_attack_type = "none"
            confidence = self._bounded_confidence(0.88 + max_plausibility * 0.08)
            verdict = "accept"
        elif max(top_sensor, top_gateway) < 2.5 and max(top_sensor_raw, top_gateway_raw) < 2.8:
            predicted_attack_type = "none"
            confidence = 0.92
            verdict = "accept"
        elif max_replay_gap >= 3.0 or replay_event_count >= 2:
            predicted_attack_type = "replay_attack"
            confidence = self._bounded_confidence(0.77 + min(max_replay_gap / 8.0, 0.18))
            verdict = "reject"
        elif max_replay_gap >= 1.0 and counter_regressions >= 10:
            predicted_attack_type = "replay_attack"
            confidence = self._bounded_confidence(0.73 + min(counter_regressions / 200.0, 0.18))
            verdict = "reject"
        elif delayed_duplicate_count >= 20 and max_duplicate_gap_s >= 30.0:
            predicted_attack_type = "replay_attack"
            confidence = self._bounded_confidence(0.73 + min(max_duplicate_gap_s / 300.0, 0.18))
            verdict = "reject"
        elif fabricated >= 5 and min_gateway_trust < 0.9 and event_score >= 1.2:
            predicted_attack_type = "gateway_fabrication"
            confidence = self._bounded_confidence(0.75 + min((1.0 - min_gateway_trust), 0.18))
            verdict = "reject"
        elif multiplicity >= 3 and min_gateway_trust < 0.85:
            predicted_attack_type = "counter_corruption"
            confidence = self._bounded_confidence(0.71 + min(event_score / 6.0, 0.16))
            verdict = "reject"
        elif sensor_ratio < 0.55 or gateway_ratio < 0.55:
            predicted_attack_type = "packet_drop"
            confidence = self._bounded_confidence(0.75 + (1.0 - min(sensor_ratio, gateway_ratio)))
            verdict = "reject"
        elif gateway_shift >= 6.0 and gateway_impacted >= 4 and gateway_shift > sensor_shift:
            predicted_attack_type = "gateway_bias"
            confidence = self._bounded_confidence(0.78 + gateway_shift / 25.0)
            verdict = "reject"
        elif sensor_shift >= 6.0 and sensor_shift > gateway_shift:
            predicted_attack_type = "sensor_foil"
            confidence = self._bounded_confidence(0.76 + sensor_shift / 25.0)
            verdict = "reject"
        elif std_delta >= 2.0 or residual_delta >= 10.0:
            predicted_attack_type = "random_noise"
            confidence = self._bounded_confidence(0.72 + max(std_delta, residual_delta / 10.0) / 10.0)
            verdict = "reject"
        elif multiplicity >= 2 and min_gateway_trust < 0.8:
            predicted_attack_type = "selective_suppression"
            confidence = self._bounded_confidence(0.70 + min(multiplicity / 10.0, 0.18))
            verdict = "reject"

        reasons: list[str] = []
        if confidence < 0.8:
            reasons.append("low_consensus_confidence")
        if verdict == "uncertain":
            reasons.append("no_strong_rule_fired")
        if predicted_attack_type == "none" and max(top_sensor, top_gateway) >= 2.5:
            reasons.append("rf_layer_still_suspicious")
        if predicted_attack_type == "none" and event_score >= 1.2:
            reasons.append("trust_layer_still_suspicious")
        return HeuristicAssessment(
            verdict="uncertain" if reasons else verdict,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            should_invoke_llm=bool(reasons),
            trigger_reasons=reasons,
            evidence=evidence,
        )

    def _localization_only_consensus(
        self,
        *,
        evidence: dict[str, Any],
    ) -> HeuristicAssessment:
        predicted_attack_type = "none"
        confidence = 0.58
        verdict = "uncertain"
        top_sensor = float(evidence["top_sensor_score"])
        top_gateway = float(evidence["top_gateway_score"])
        sensor_ratio = float(evidence["sensor_worst_packet_ratio"])
        gateway_ratio = float(evidence["gateway_worst_packet_ratio"])
        sensor_shift = float(evidence["sensor_mean_abs_rssi_shift_db"])
        gateway_shift = float(evidence["gateway_mean_abs_rssi_shift_db"])
        gateway_impacted = int(evidence["gateway_impacted_sensor_count"])
        std_delta = float(evidence["sensor_mean_std_delta_db"])
        residual_delta = float(evidence["trilateration_residual_delta_m"])
        localization_delta = float(evidence.get("localization_error_delta_m", 0.0))

        if max(top_sensor, top_gateway) < 2.5 and residual_delta < 8.0 and localization_delta < 20.0:
            predicted_attack_type = "none"
            confidence = 0.90
            verdict = "accept"
        elif sensor_ratio < 0.55 or gateway_ratio < 0.55:
            predicted_attack_type = "packet_drop"
            confidence = self._bounded_confidence(0.73 + (1.0 - min(sensor_ratio, gateway_ratio)))
            verdict = "reject"
        elif gateway_shift >= 6.0 and gateway_impacted >= 4 and gateway_shift > sensor_shift:
            predicted_attack_type = "gateway_bias"
            confidence = self._bounded_confidence(0.77 + gateway_shift / 25.0)
            verdict = "reject"
        elif sensor_shift >= 6.0 and sensor_shift > gateway_shift:
            predicted_attack_type = "sensor_foil"
            confidence = self._bounded_confidence(0.76 + sensor_shift / 25.0)
            verdict = "reject"
        elif std_delta >= 2.0 or residual_delta >= 10.0 or localization_delta >= 30.0:
            predicted_attack_type = "random_noise"
            confidence = self._bounded_confidence(
                0.71 + max(std_delta, residual_delta / 10.0, localization_delta / 30.0) / 10.0
            )
            verdict = "reject"

        reasons: list[str] = []
        if confidence < 0.8:
            reasons.append("low_localization_confidence")
        if verdict == "uncertain":
            reasons.append("no_strong_rf_rule_fired")
        if predicted_attack_type == "none" and max(top_sensor, top_gateway) >= 2.5:
            reasons.append("rf_layer_still_suspicious")
        return HeuristicAssessment(
            verdict="uncertain" if reasons else verdict,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            should_invoke_llm=bool(reasons),
            trigger_reasons=reasons,
            evidence=evidence,
        )

    @staticmethod
    def _select_llm_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "sensor_mean_abs_rssi_shift_db",
            "sensor_worst_packet_ratio",
            "sensor_mean_std_delta_db",
            "sensor_expected_extra_attenuation_db",
            "sensor_environment_plausibility",
            "gateway_mean_abs_rssi_shift_db",
            "gateway_worst_packet_ratio",
            "gateway_impacted_sensor_count",
            "gateway_expected_extra_attenuation_db",
            "gateway_environment_plausibility",
            "localization_error_delta_m",
            "trilateration_residual_delta_m",
            "top_sensor_score_raw",
            "top_sensor_score",
            "top_gateway_score_raw",
            "top_gateway_score",
            "max_replay_gap",
            "replay_event_count",
            "counter_regression_count",
            "high_skew_witness_count",
            "fabricated_witness_count",
            "multiplicity_anomaly_count",
            "min_gateway_trust",
            "event_inconsistency_score",
            "loramas_vote_margin",
            "loramas_top_label_score",
            "loramas_runner_up_score",
            "loramas_supporting_agents",
        ]
        return {key: evidence[key] for key in keys if key in evidence}

    @staticmethod
    def _compact_path_context(frame) -> dict[str, Any]:
        if frame is None or frame.empty:
            return {}
        payload: dict[str, Any] = {
            "link_count": int(frame.shape[0]),
        }
        for column in [
            "path_distance_m",
            "building_intersections",
            "vegetation_intersections",
            "line_of_sight_blocked",
            "expected_extra_attenuation_db",
            "context_fragility_score",
        ]:
            if column in frame.columns:
                payload[f"mean_{column}"] = ReActGatewayAgent._json_safe(float(frame[column].mean()))
        return payload

    @staticmethod
    def _compact_trilateration_delta(
        *,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "available": bool(trilateration.get("available")) and bool(baseline_trilateration.get("available")),
        }
        if not payload["available"]:
            payload["reason"] = trilateration.get("reason") or baseline_trilateration.get("reason")
            return payload
        current_error = float(trilateration.get("error_m", np.nan))
        baseline_error = float(baseline_trilateration.get("error_m", np.nan))
        current_residual = float(trilateration.get("residual_rmse_m", np.nan))
        baseline_residual = float(baseline_trilateration.get("residual_rmse_m", np.nan))
        payload["error_delta_m"] = ReActGatewayAgent._json_safe(current_error - baseline_error)
        payload["residual_delta_m"] = ReActGatewayAgent._json_safe(current_residual - baseline_residual)
        payload["pair_count"] = ReActGatewayAgent._json_safe(trilateration.get("pair_count"))
        return payload

    @staticmethod
    def _compact_frame(frame, limit: int) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        payload = frame.head(limit).to_dict(orient="records")
        return [{key: ReActGatewayAgent._json_safe(value) for key, value in row.items()} for row in payload]

    @staticmethod
    def _compact_trilateration(payload: dict[str, Any]) -> dict[str, Any]:
        keep = [
            "available",
            "sensor",
            "error_m",
            "residual_rmse_m",
            "pair_count",
            "x_est",
            "y_est",
            "reason",
        ]
        return {key: ReActGatewayAgent._json_safe(payload.get(key)) for key in keep if key in payload}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _top_score(rank: Any) -> float:
        if rank is None or getattr(rank, "empty", True):
            return 0.0
        return float(rank.iloc[0]["anomaly_score"])

    @staticmethod
    def _choose_sensor(sensor_rank: Any, scenario: AttackScenario) -> str | None:
        if scenario.sensor:
            return scenario.sensor
        if scenario.sensors:
            return str(scenario.sensors[0])
        if sensor_rank is None or sensor_rank.empty:
            return None
        return str(sensor_rank.iloc[0]["sensor"])

    @staticmethod
    def _choose_gateway(gateway_rank: Any, scenario: AttackScenario) -> str | None:
        if scenario.gateway:
            return scenario.gateway
        if scenario.gateways:
            return str(scenario.gateways[0])
        if gateway_rank is None or gateway_rank.empty:
            return None
        return str(gateway_rank.iloc[0]["gateway"])

    @staticmethod
    def _summarize_sensor_view(sensor_view) -> str:
        mean_shift = float(sensor_view["rssi_shift_db"].mean())
        mean_abs_shift = float(sensor_view["rssi_shift_db"].abs().mean())
        min_ratio = float(sensor_view["packet_ratio"].min())
        mean_std_delta = float(sensor_view["std_delta_db"].mean())
        summary = (
            f"Across {int(sensor_view.shape[0])} gateways, the sensor shows mean RSSI shift {mean_shift:.2f} dB "
            f"(mean absolute {mean_abs_shift:.2f} dB), worst packet-retention ratio {min_ratio:.2f}, "
            f"and mean RSSI spread change {mean_std_delta:.2f} dB."
        )
        if "context_fragility_score" in sensor_view.columns:
            summary += (
                f" The path context has mean fragility {float(sensor_view['context_fragility_score'].mean()):.2f}, "
                f"with {float(sensor_view['line_of_sight_blocked'].mean()):.2f} of links crossing satellite-derived obstructions "
                f"and expected extra attenuation {float(sensor_view['expected_extra_attenuation_db'].mean()):.2f} dB."
            )
        return summary

    @staticmethod
    def _summarize_gateway_view(gateway_view) -> str:
        mean_shift = float(gateway_view["rssi_shift_db"].mean())
        mean_abs_shift = float(gateway_view["rssi_shift_db"].abs().mean())
        impacted = int((gateway_view["rssi_shift_db"].abs() >= 6.0).sum())
        min_ratio = float(gateway_view["packet_ratio"].min())
        summary = (
            f"This gateway affects {int(gateway_view.shape[0])} sensors with mean RSSI shift {mean_shift:.2f} dB "
            f"(mean absolute {mean_abs_shift:.2f} dB); {impacted} sensors exceed the 6 dB shift threshold and "
            f"the worst packet-retention ratio is {min_ratio:.2f}."
        )
        if "context_fragility_score" in gateway_view.columns:
            summary += (
                f" The satellite-derived path context has mean fragility {float(gateway_view['context_fragility_score'].mean()):.2f} "
                f"and obstruction coverage {float(gateway_view['line_of_sight_blocked'].mean()):.2f}, "
                f"with expected extra attenuation {float(gateway_view['expected_extra_attenuation_db'].mean()):.2f} dB."
            )
        return summary

    @staticmethod
    def _summarize_trilateration(current: dict[str, Any], baseline: dict[str, Any]) -> str:
        if not current.get("available"):
            return current.get("reason", "Trilateration was unavailable.")

        baseline_error = float(baseline.get("error_m", np.nan)) if baseline.get("available") else np.nan
        current_error = float(current.get("error_m", np.nan))
        baseline_residual = (
            float(baseline.get("residual_rmse_m", np.nan)) if baseline.get("available") else np.nan
        )
        current_residual = float(current.get("residual_rmse_m", np.nan))
        return (
            f"Estimated location ({current['x_est']:.1f}, {current['y_est']:.1f}) m using {current['pair_count']} gateways. "
            f"Localization error changed from {baseline_error:.2f} m to {current_error:.2f} m and the multilateration residual "
            f"changed from {baseline_residual:.2f} m to {current_residual:.2f} m."
        )

    def _build_evidence(
        self,
        sensor_view,
        gateway_view,
        sensor_rank,
        gateway_rank,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
        trust: dict[str, Any],
    ) -> dict[str, Any]:
        sensor_shift = self._safe_mean_abs(sensor_view, "rssi_shift_db")
        sensor_ratio = self._safe_min(sensor_view, "packet_ratio", default=1.0)
        sensor_std_delta = self._safe_mean(sensor_view, "std_delta_db")

        gateway_shift = self._safe_mean_abs(gateway_view, "rssi_shift_db")
        gateway_ratio = self._safe_min(gateway_view, "packet_ratio", default=1.0)
        gateway_impacted = (
            int((gateway_view["rssi_shift_db"].abs() >= 6.0).sum())
            if gateway_view is not None and not gateway_view.empty
            else 0
        )
        gateway_packet_ratio_std = self._safe_std(gateway_view, "packet_ratio")
        gateway_low_ratio_sensor_count = (
            int((gateway_view["packet_ratio"] < 0.7).sum())
            if gateway_view is not None and not gateway_view.empty
            else 0
        )
        gateway_sensor_count = int(gateway_view.shape[0]) if gateway_view is not None and not gateway_view.empty else 0
        sensor_fragility = self._safe_mean(sensor_view, "context_fragility_score")
        sensor_blocked_fraction = self._safe_mean(sensor_view, "line_of_sight_blocked")
        gateway_fragility = self._safe_mean(gateway_view, "context_fragility_score")
        gateway_blocked_fraction = self._safe_mean(gateway_view, "line_of_sight_blocked")
        sensor_building_hits = self._safe_mean(sensor_view, "building_intersections")
        gateway_building_hits = self._safe_mean(gateway_view, "building_intersections")
        sensor_expected_attenuation = self._safe_mean(sensor_view, "expected_extra_attenuation_db")
        gateway_expected_attenuation = self._safe_mean(gateway_view, "expected_extra_attenuation_db")
        sensor_plausibility = self._environment_plausibility(sensor_shift, sensor_expected_attenuation, sensor_ratio)
        gateway_plausibility = self._environment_plausibility(gateway_shift, gateway_expected_attenuation, gateway_ratio)

        baseline_error = float(baseline_trilateration.get("error_m", np.nan))
        current_error = float(trilateration.get("error_m", np.nan))
        baseline_residual = float(baseline_trilateration.get("residual_rmse_m", np.nan))
        current_residual = float(trilateration.get("residual_rmse_m", np.nan))

        error_delta = (
            current_error - baseline_error
            if np.isfinite(current_error) and np.isfinite(baseline_error)
            else 0.0
        )
        residual_delta = (
            current_residual - baseline_residual
            if np.isfinite(current_residual) and np.isfinite(baseline_residual)
            else 0.0
        )
        min_gateway_trust = min(trust.get("gateway_trust_scores", {}).values()) if trust.get("gateway_trust_scores") else 1.0
        max_replay_gap = float(trust.get("max_replay_gap", 0.0))
        replay_event_count = int(trust.get("replay_event_count", 0))
        fabricated_witness_count = int(trust.get("fabricated_witness_count", 0))
        multiplicity_anomaly_count = int(trust.get("multiplicity_anomaly_count", 0))
        event_inconsistency_score = float(trust.get("event_inconsistency_score", 0.0))

        evidence = {
            "sensor_mean_abs_rssi_shift_db": round(sensor_shift, 3),
            "sensor_worst_packet_ratio": round(sensor_ratio, 3),
            "sensor_mean_std_delta_db": round(sensor_std_delta, 3),
            "gateway_mean_abs_rssi_shift_db": round(gateway_shift, 3),
            "gateway_worst_packet_ratio": round(gateway_ratio, 3),
            "gateway_impacted_sensor_count": gateway_impacted,
            "gateway_packet_ratio_std": round(gateway_packet_ratio_std, 3),
            "gateway_low_ratio_sensor_count": gateway_low_ratio_sensor_count,
            "gateway_sensor_count": gateway_sensor_count,
            "localization_error_delta_m": round(error_delta, 3),
            "trilateration_residual_delta_m": round(residual_delta, 3),
            "top_sensor_score": round(self._top_score(sensor_rank), 3),
            "top_gateway_score": round(self._top_score(gateway_rank), 3),
            "sensor_context_fragility": round(sensor_fragility, 3),
            "sensor_blocked_link_fraction": round(sensor_blocked_fraction, 3),
            "sensor_mean_building_intersections": round(sensor_building_hits, 3),
            "sensor_expected_extra_attenuation_db": round(sensor_expected_attenuation, 3),
            "sensor_environment_plausibility": round(sensor_plausibility, 3),
            "gateway_context_fragility": round(gateway_fragility, 3),
            "gateway_blocked_link_fraction": round(gateway_blocked_fraction, 3),
            "gateway_mean_building_intersections": round(gateway_building_hits, 3),
            "gateway_expected_extra_attenuation_db": round(gateway_expected_attenuation, 3),
            "gateway_environment_plausibility": round(gateway_plausibility, 3),
            "max_replay_gap": round(max_replay_gap, 3),
            "replay_event_count": replay_event_count,
            "counter_regression_count": int(trust.get("counter_regression_count", 0)),
            "delayed_duplicate_count": int(trust.get("delayed_duplicate_count", 0)),
            "max_duplicate_gap_s": round(float(trust.get("max_duplicate_gap_s", 0.0)), 3),
            "high_skew_witness_count": int(trust.get("high_skew_witness_count", 0)),
            "fabricated_witness_count": fabricated_witness_count,
            "multiplicity_anomaly_count": multiplicity_anomaly_count,
            "min_gateway_trust": round(float(min_gateway_trust), 3),
            "event_inconsistency_score": round(event_inconsistency_score, 3),
        }
        if sensor_rank is not None and not getattr(sensor_rank, "empty", True) and "base_anomaly_score" in sensor_rank.columns:
            evidence["top_sensor_score_raw"] = round(float(sensor_rank.iloc[0]["base_anomaly_score"]), 3)
        if gateway_rank is not None and not getattr(gateway_rank, "empty", True) and "base_anomaly_score" in gateway_rank.columns:
            evidence["top_gateway_score_raw"] = round(float(gateway_rank.iloc[0]["base_anomaly_score"]), 3)

        return evidence

    @staticmethod
    def _summarize_trust(trust: dict[str, Any]) -> str:
        gateway_scores = trust.get("gateway_trust_scores", {})
        trust_bits = ", ".join(f"{gw}={score:.2f}" for gw, score in gateway_scores.items()) or "no trust scores"
        return (
            f"Packet witness analysis used a {trust.get('time_tolerance_s', 0.0):.2f} s timestamp tolerance. "
            f"Max replay gap is {trust.get('max_replay_gap', 0.0):.2f}, "
            f"delayed duplicate witness count is {trust.get('delayed_duplicate_count', 0)}, "
            f"{trust.get('high_skew_witness_count', 0)} witness sets exceeded the skew bound, "
            f"{trust.get('fabricated_witness_count', 0)} looked fabricated, and "
            f"{trust.get('multiplicity_anomaly_count', 0)} showed multiplicity anomalies. "
            f"Gateway trust scores: {trust_bits}."
        )

    @staticmethod
    def _summarize_environment_evidence(evidence: dict[str, Any]) -> str:
        return (
            f"Sensor path fragility is {evidence.get('sensor_context_fragility', 0.0):.2f} with blocked-link fraction "
            f"{evidence.get('sensor_blocked_link_fraction', 0.0):.2f} and expected extra attenuation "
            f"{evidence.get('sensor_expected_extra_attenuation_db', 0.0):.2f} dB. Gateway path fragility is "
            f"{evidence.get('gateway_context_fragility', 0.0):.2f} with blocked-link fraction "
            f"{evidence.get('gateway_blocked_link_fraction', 0.0):.2f}. Environment plausibility scores are "
            f"sensor={evidence.get('sensor_environment_plausibility', 0.0):.2f} and "
            f"gateway={evidence.get('gateway_environment_plausibility', 0.0):.2f}."
        )

    @staticmethod
    def _environment_plausibility(shift_db: float, expected_extra_attenuation_db: float, packet_ratio: float) -> float:
        if expected_extra_attenuation_db <= 0.0:
            return 0.0
        shift_alignment = min(abs(shift_db) / max(expected_extra_attenuation_db, 1.0), 1.0)
        packet_consistency = 1.0 if packet_ratio >= 0.9 else max((packet_ratio - 0.6) / 0.3, 0.0)
        return float(np.clip(0.65 * shift_alignment + 0.35 * packet_consistency, 0.0, 1.0))

    @staticmethod
    def _safe_mean_abs(df, column: str) -> float:
        if df is None or df.empty:
            return 0.0
        return float(df[column].abs().mean())

    @staticmethod
    def _safe_mean(df, column: str) -> float:
        if df is None or df.empty or column not in df.columns:
            return 0.0
        return float(df[column].mean())

    @staticmethod
    def _safe_std(df, column: str) -> float:
        if df is None or df.empty or column not in df.columns:
            return 0.0
        return float(df[column].std(ddof=0))

    @staticmethod
    def _safe_min(df, column: str, default: float) -> float:
        if df is None or df.empty or column not in df.columns:
            return default
        return float(df[column].min())

    @staticmethod
    def _bounded_confidence(value: float) -> float:
        return float(np.clip(value, 0.0, 0.99))
