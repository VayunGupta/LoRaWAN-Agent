from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import AgentClaim, HeuristicAssessment


@dataclass
class FactorContribution:
    label: str
    source: str
    weight: float


class ConsistencyGraphVerifier:
    """
    A more principled verifier than the original weighted vote supervisor.

    The verifier treats attack labels as latent event hypotheses and combines:
    - unary evidence from each specialist claim
    - typed consistency factors from the shared evidence dictionary
    - pairwise support edges between compatible specialist claims

    This is still lightweight, but it makes the aggregation rule explicit and
    inspectable instead of hiding it in one flat weighted sum.
    """

    LABELS = [
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

    ROLE_RELIABILITY = {
        "gateway_local": 1.00,
        "temporal_consistency": 1.25,
        "physical_consistency": 1.15,
    }

    SAME_LABEL_EDGE_BONUS = {
        frozenset({"temporal_consistency", "gateway_local"}): 0.18,
        frozenset({"physical_consistency", "gateway_local"}): 0.20,
    }

    def verify(
        self,
        *,
        agent_claims: list[AgentClaim],
        evidence: dict[str, Any],
        centralized: HeuristicAssessment,
    ) -> HeuristicAssessment:
        label_scores = {label: 0.0 for label in self.LABELS}
        contributions: list[FactorContribution] = []
        label_support: dict[str, list[str]] = {label: [] for label in self.LABELS}

        for claim in agent_claims:
            weight = self._claim_weight(claim)
            label_scores[claim.label] += weight
            label_support.setdefault(claim.label, []).append(claim.agent_name)
            contributions.append(
                FactorContribution(
                    label=claim.label,
                    source=f"claim:{claim.agent_name}",
                    weight=weight,
                )
            )
            if claim.label == "none":
                # Explicit benign votes should still regularize attack families.
                for attack_label in self.LABELS:
                    if attack_label == "none":
                        continue
                    penalty = 0.10 * weight
                    label_scores[attack_label] -= penalty
                    contributions.append(
                        FactorContribution(
                            label=attack_label,
                            source=f"benign_penalty:{claim.agent_name}",
                            weight=-penalty,
                        )
                    )

        self._apply_pairwise_edges(
            agent_claims=agent_claims,
            label_scores=label_scores,
            contributions=contributions,
        )
        self._apply_evidence_factors(
            evidence=evidence,
            label_scores=label_scores,
            contributions=contributions,
        )
        self._apply_claim_context_overrides(
            agent_claims=agent_claims,
            evidence=evidence,
            label_scores=label_scores,
            contributions=contributions,
        )

        # Keep the strong centralized baseline as a backstop rather than the main method.
        if centralized.predicted_attack_type in label_scores and centralized.predicted_attack_type != "none":
            bonus = 0.30 * float(centralized.confidence)
            label_scores[centralized.predicted_attack_type] += bonus
            contributions.append(
                FactorContribution(
                    label=centralized.predicted_attack_type,
                    source="centralized_backstop",
                    weight=bonus,
                )
            )
        else:
            benign_bonus = 0.15 * float(centralized.confidence)
            label_scores["none"] += benign_bonus
            contributions.append(
                FactorContribution(
                    label="none",
                    source="centralized_backstop",
                    weight=benign_bonus,
                )
            )

        ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_attack_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score

        confidence = self._bounded_confidence(
            0.50 + min(max(top_score, 0.0) / 4.5, 0.22) + min(max(margin, 0.0) / 2.0, 0.22)
        )
        verdict = "accept" if predicted_attack_type == "none" else "reject"
        trigger_reasons: list[str] = []

        if margin < 0.30:
            trigger_reasons.append("graph_low_margin")
        if predicted_attack_type != "none" and len(label_support.get(predicted_attack_type, [])) < 2:
            trigger_reasons.append("graph_sparse_support")
        if predicted_attack_type == "none" and float(evidence.get("event_inconsistency_score", 0.0)) >= 1.2:
            trigger_reasons.append("graph_unresolved_event_inconsistency")
        if (
            predicted_attack_type != "replay_attack"
            and float(evidence.get("max_replay_gap", 0.0)) >= 2.0
        ):
            trigger_reasons.append("graph_temporal_conflict")
        if predicted_attack_type == "none" and max(
            float(evidence.get("top_sensor_score", 0.0)),
            float(evidence.get("top_gateway_score", 0.0)),
        ) >= 2.5:
            trigger_reasons.append("graph_rf_conflict")
        if trigger_reasons:
            verdict = "uncertain"

        top_factors = self._top_factors(contributions, predicted_attack_type)
        enriched_evidence = dict(evidence)
        enriched_evidence["consistency_graph_top_label_score"] = round(float(top_score), 3)
        enriched_evidence["consistency_graph_runner_up_score"] = round(float(second_score), 3)
        enriched_evidence["consistency_graph_margin"] = round(float(margin), 3)
        enriched_evidence["consistency_graph_supporting_agents"] = label_support.get(predicted_attack_type, [])
        enriched_evidence["consistency_graph_top_factors"] = top_factors

        return HeuristicAssessment(
            verdict=verdict,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            should_invoke_llm=bool(trigger_reasons),
            trigger_reasons=trigger_reasons,
            evidence=enriched_evidence,
        )

    def _apply_pairwise_edges(
        self,
        *,
        agent_claims: list[AgentClaim],
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
    ) -> None:
        for i, left in enumerate(agent_claims):
            for right in agent_claims[i + 1 :]:
                if left.label != right.label:
                    continue
                pair_key = frozenset({left.role, right.role})
                bonus = self.SAME_LABEL_EDGE_BONUS.get(pair_key, 0.08)
                if left.label == "none":
                    bonus *= 0.6
                label_scores[left.label] += bonus
                contributions.append(
                    FactorContribution(
                        label=left.label,
                        source=f"edge:{left.role}<->{right.role}",
                        weight=bonus,
                    )
                )

    def _apply_evidence_factors(
        self,
        *,
        evidence: dict[str, Any],
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
    ) -> None:
        sensor_shift = float(evidence.get("sensor_mean_abs_rssi_shift_db", 0.0))
        gateway_shift = float(evidence.get("gateway_mean_abs_rssi_shift_db", 0.0))
        sensor_ratio = float(evidence.get("sensor_worst_packet_ratio", 1.0))
        gateway_ratio = float(evidence.get("gateway_worst_packet_ratio", 1.0))
        std_delta = float(evidence.get("sensor_mean_std_delta_db", 0.0))
        residual_delta = float(evidence.get("trilateration_residual_delta_m", 0.0))
        gateway_impacted = int(evidence.get("gateway_impacted_sensor_count", 0))
        max_replay_gap = float(evidence.get("max_replay_gap", 0.0))
        replay_events = int(evidence.get("replay_event_count", 0))
        regressions = int(evidence.get("counter_regression_count", 0))
        delayed_duplicates = int(evidence.get("delayed_duplicate_count", 0))
        duplicate_gap = float(evidence.get("max_duplicate_gap_s", 0.0))
        fabricated = int(evidence.get("fabricated_witness_count", 0))
        multiplicity = int(evidence.get("multiplicity_anomaly_count", 0))
        min_gateway_trust = float(evidence.get("min_gateway_trust", 1.0))
        event_score = float(evidence.get("event_inconsistency_score", 0.0))
        environment_plausibility = max(
            float(evidence.get("sensor_environment_plausibility", 0.0)),
            float(evidence.get("gateway_environment_plausibility", 0.0)),
        )
        top_sensor = float(evidence.get("top_sensor_score", 0.0))
        top_gateway = float(evidence.get("top_gateway_score", 0.0))

        self._add_factor(label_scores, contributions, "replay_attack", "factor:temporal_gap", 0.55 * min(max_replay_gap, 4.0))
        self._add_factor(label_scores, contributions, "replay_attack", "factor:replay_events", 0.24 * min(replay_events, 3))
        self._add_factor(label_scores, contributions, "replay_attack", "factor:counter_regressions", 0.02 * min(regressions, 20))

        self._add_factor(label_scores, contributions, "counter_corruption", "factor:multiplicity", 0.28 * min(multiplicity, 4))
        self._add_factor(label_scores, contributions, "counter_corruption", "factor:trust_drop", 1.8 * max(0.0, 0.95 - min_gateway_trust))
        self._add_factor(label_scores, contributions, "counter_corruption", "factor:event_inconsistency", 0.18 * min(event_score, 4.0))

        self._add_factor(label_scores, contributions, "gateway_fabrication", "factor:fabrication", 0.22 * min(fabricated, 6))
        self._add_factor(label_scores, contributions, "gateway_fabrication", "factor:gateway_trust", 2.0 * max(0.0, 0.92 - min_gateway_trust))
        self._add_factor(label_scores, contributions, "gateway_fabrication", "factor:gateway_event", 0.16 * min(event_score, 4.0))

        self._add_factor(
            label_scores,
            contributions,
            "gateway_bias",
            "factor:gateway_shift",
            0.16 * max(0.0, gateway_shift - sensor_shift / 2.0),
        )
        self._add_factor(
            label_scores,
            contributions,
            "gateway_bias",
            "factor:gateway_impacted",
            0.10 * min(gateway_impacted, 5),
        )

        self._add_factor(
            label_scores,
            contributions,
            "sensor_foil",
            "factor:sensor_shift",
            0.16 * max(0.0, sensor_shift - gateway_shift / 2.0),
        )

        packet_drop_support = 2.2 * max(0.0, 0.8 - min(sensor_ratio, gateway_ratio))
        self._add_factor(label_scores, contributions, "packet_drop", "factor:packet_ratio", packet_drop_support)
        self._add_factor(label_scores, contributions, "selective_suppression", "factor:packet_ratio", 0.7 * packet_drop_support)

        self._add_factor(
            label_scores,
            contributions,
            "random_noise",
            "factor:variance",
            0.24 * min(max(std_delta, residual_delta / 10.0), 4.0),
        )

        benign_support = 0.0
        if max(top_sensor, top_gateway) < 2.5:
            benign_support += 0.85
        benign_support += 0.55 * environment_plausibility
        if event_score < 1.0:
            benign_support += 0.40
        self._add_factor(label_scores, contributions, "none", "factor:benign_prior", benign_support)

        if environment_plausibility >= 0.65:
            for label in ("sensor_foil", "gateway_bias", "random_noise"):
                self._add_factor(label_scores, contributions, label, "penalty:environment_plausibility", -0.30)
        if max_replay_gap >= 2.0:
            self._add_factor(label_scores, contributions, "none", "penalty:temporal_conflict", -0.55)
        if min_gateway_trust < 0.85:
            self._add_factor(label_scores, contributions, "none", "penalty:trust_conflict", -0.35)

    def _claim_weight(self, claim: AgentClaim) -> float:
        base = self.ROLE_RELIABILITY.get(claim.role, 1.0) * max(float(claim.confidence), 0.0)
        if claim.label == "none":
            return 0.45 * base
        return base

    def _apply_claim_context_overrides(
        self,
        *,
        agent_claims: list[AgentClaim],
        evidence: dict[str, Any],
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
    ) -> None:
        temporal_claim = next((claim for claim in agent_claims if claim.role == "temporal_consistency"), None)
        if temporal_claim and temporal_claim.label == "replay_attack":
            rf_quiet = max(
                float(evidence.get("top_sensor_score", 0.0)),
                float(evidence.get("top_gateway_score", 0.0)),
            ) < 2.0
            regressions = int(evidence.get("counter_regression_count", 0))
            replay_gap = float(evidence.get("max_replay_gap", 0.0))
            if rf_quiet and (regressions >= 50 or replay_gap >= 1.0):
                bonus = 1.25 + min(regressions / 200.0, 0.75) + 0.4 * min(replay_gap, 3.0)
                self._add_factor(
                    label_scores,
                    contributions,
                    "replay_attack",
                    "override:temporal_strong_under_quiet_rf",
                    bonus,
                )
                self._add_factor(
                    label_scores,
                    contributions,
                    "none",
                    "override:temporal_strong_under_quiet_rf",
                    -0.85,
                )

    @staticmethod
    def _add_factor(
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
        label: str,
        source: str,
        weight: float,
    ) -> None:
        label_scores[label] += weight
        contributions.append(FactorContribution(label=label, source=source, weight=weight))

    @staticmethod
    def _top_factors(contributions: list[FactorContribution], label: str) -> list[str]:
        top = sorted(
            [item for item in contributions if item.label == label],
            key=lambda item: item.weight,
            reverse=True,
        )[:5]
        return [f"{item.source}:{round(float(item.weight), 3)}" for item in top]

    @staticmethod
    def _bounded_confidence(value: float) -> float:
        return float(np.clip(value, 0.05, 0.99))


class EnergyConsistencyVerifier:
    """
    Structured energy-style verifier over normalized evidence features.

    Compared with the lighter consistency graph, this verifier makes the
    label-evidence interaction explicit: each attack label has a template over
    temporal, physical, trust, availability, and benignity features. Agent
    claims contribute support to labels, but the final decision comes from the
    total label energy rather than a flat vote.
    """

    LABELS = ConsistencyGraphVerifier.LABELS
    FEATURE_NAMES = [
        "rf_quiet",
        "environment_plausibility",
        "event_inconsistency",
        "low_event_inconsistency",
        "trust_drop",
        "low_trust_drop",
        "packet_loss",
        "low_packet_loss",
        "sensor_shift_dominance",
        "gateway_shift_dominance",
        "variance",
        "gateway_impacted",
        "temporal_gap",
        "replay_events",
        "counter_regressions",
        "delayed_duplicates",
        "duplicate_gap",
        "fabricated_witnesses",
        "multiplicity",
    ]
    ROLE_RELIABILITY = {
        "gateway_local": 0.95,
        "temporal_consistency": 1.20,
        "physical_consistency": 1.15,
    }
    DEFAULT_LABEL_FEATURE_WEIGHTS = {
        "none": {
            "rf_quiet": 1.20,
            "environment_plausibility": 0.75,
            "low_event_inconsistency": 0.80,
            "low_trust_drop": 0.55,
            "low_packet_loss": 0.35,
            "temporal_gap": -1.10,
            "event_inconsistency": -0.70,
            "trust_drop": -0.65,
            "packet_loss": -0.45,
            "sensor_shift_dominance": -0.90,
            "gateway_shift_dominance": -0.90,
            "variance": -0.80,
            "counter_regressions": -1.10,
            "fabricated_witnesses": -1.00,
            "multiplicity": -0.85,
        },
        "sensor_foil": {
            "sensor_shift_dominance": 1.20,
            "environment_plausibility": -0.35,
            "variance": 0.20,
        },
        "gateway_bias": {
            "gateway_shift_dominance": 1.15,
            "gateway_impacted": 0.80,
            "environment_plausibility": -0.20,
        },
        "random_noise": {
            "variance": 1.25,
            "rf_quiet": -0.25,
            "sensor_shift_dominance": -0.75,
            "gateway_shift_dominance": -0.75,
        },
        "packet_drop": {
            "packet_loss": 1.20,
            "event_inconsistency": 0.25,
            "trust_drop": 0.15,
        },
        "replay_attack": {
            "temporal_gap": 1.35,
            "replay_events": 1.00,
            "counter_regressions": 0.75,
            "delayed_duplicates": 1.10,
            "duplicate_gap": 1.10,
            "rf_quiet": 0.35,
            "packet_loss": -0.20,
        },
        "gateway_fabrication": {
            "fabricated_witnesses": 1.25,
            "trust_drop": 1.10,
            "event_inconsistency": 0.85,
            "gateway_impacted": 0.30,
        },
        "counter_corruption": {
            "multiplicity": 1.10,
            "trust_drop": 1.00,
            "event_inconsistency": 0.95,
            "counter_regressions": 0.45,
            "delayed_duplicates": -0.60,
            "duplicate_gap": -0.60,
        },
        "selective_suppression": {
            "packet_loss": 0.90,
            "event_inconsistency": 0.75,
            "trust_drop": 0.45,
            "gateway_impacted": 0.25,
        },
    }

    def __init__(self) -> None:
        self.label_feature_weights = {
            label: dict(weights) for label, weights in self.DEFAULT_LABEL_FEATURE_WEIGHTS.items()
        }
        self.label_bias = {label: 0.0 for label in self.LABELS}
        self.fitted = False
        self.tunable_params = {
            "bias_scale": 1.0,
            "claim_scale": 1.0,
            "none_claim_discount": 0.5,
            "agreement_scale": 1.0,
            "template_scale": 1.0,
            "contradiction_scale": 1.0,
            "centralized_scale": 0.25,
            "none_label_offset": 0.0,
            "confidence_trigger_threshold": 0.48,
            "margin_trigger_threshold": 0.25,
        }

    def get_tunable_params(self) -> dict[str, float]:
        return dict(self.tunable_params)

    def set_tunable_params(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if key in self.tunable_params:
                self.tunable_params[key] = float(value)

    def fit(self, examples: list[dict[str, Any]]) -> None:
        if not examples:
            return

        labels: list[str] = []
        feature_rows: list[list[float]] = []
        for example in examples:
            label = str(example.get("label", "none"))
            evidence = dict(example.get("evidence", {}))
            features = self._extract_features(evidence)
            labels.append(label)
            feature_rows.append([float(features.get(name, 0.0)) for name in self.FEATURE_NAMES])

        if not feature_rows:
            return

        x = np.asarray(feature_rows, dtype=float)
        all_labels = np.asarray(labels, dtype=object)
        total = int(x.shape[0])
        feature_std = np.std(x, axis=0) + 0.15
        global_mean = np.mean(x, axis=0)

        learned_weights: dict[str, dict[str, float]] = {}
        learned_bias: dict[str, float] = {}
        label_count = len(self.LABELS)

        for label in self.LABELS:
            default_weights = self.DEFAULT_LABEL_FEATURE_WEIGHTS.get(label, {})
            positive_mask = all_labels == label
            positive_count = int(np.sum(positive_mask))
            prior = np.log((positive_count + 1.0) / (total + label_count))

            if positive_count == 0:
                learned_bias[label] = -0.25
                learned_weights[label] = dict(default_weights)
                continue

            learned_bias[label] = float(np.clip(prior, -1.2, 0.5))

            pos = x[positive_mask]
            neg = x[~positive_mask] if positive_count < total else np.tile(global_mean, (1, 1))
            pos_mean = np.mean(pos, axis=0)
            neg_mean = np.mean(neg, axis=0)
            effect = (pos_mean - neg_mean) / feature_std
            shrink = positive_count / (positive_count + 3.0)

            label_weights: dict[str, float] = {}
            for idx, feature_name in enumerate(self.FEATURE_NAMES):
                default_value = float(default_weights.get(feature_name, 0.0))
                learned_value = float(np.clip(effect[idx], -2.0, 2.0))
                blended = (1.0 - shrink) * default_value + shrink * learned_value
                if abs(blended) >= 0.02:
                    label_weights[feature_name] = blended
            learned_weights[label] = label_weights

        self.label_feature_weights = learned_weights
        self.label_bias = learned_bias
        self.fitted = True

    def verify(
        self,
        *,
        agent_claims: list[AgentClaim],
        evidence: dict[str, Any],
        centralized: HeuristicAssessment,
    ) -> HeuristicAssessment:
        features = self._extract_features(evidence)
        label_scores = {label: 0.0 for label in self.LABELS}
        contributions: list[FactorContribution] = []
        label_support: dict[str, list[str]] = {label: [] for label in self.LABELS}

        for label, bias in self.label_bias.items():
            if abs(float(bias)) < 1e-6:
                continue
            scaled_bias = self.tunable_params["bias_scale"] * float(bias)
            label_scores[label] += scaled_bias
            contributions.append(
                FactorContribution(
                    label=label,
                    source="bias:label_prior",
                    weight=scaled_bias,
                )
            )

        if abs(self.tunable_params["none_label_offset"]) > 1e-6:
            label_scores["none"] += self.tunable_params["none_label_offset"]
            contributions.append(
                FactorContribution(
                    label="none",
                    source="bias:none_label_offset",
                    weight=self.tunable_params["none_label_offset"],
                )
            )

        for claim in agent_claims:
            base = self.ROLE_RELIABILITY.get(claim.role, 1.0) * max(float(claim.confidence), 0.0)
            base *= self.tunable_params["claim_scale"]
            if claim.label == "none":
                base *= self.tunable_params["none_claim_discount"]
            label_scores[claim.label] += base
            label_support[claim.label].append(claim.agent_name)
            contributions.append(
                FactorContribution(
                    label=claim.label,
                    source=f"claim:{claim.agent_name}",
                    weight=base,
                )
            )

        self._apply_claim_edges(agent_claims, label_scores, contributions)
        self._apply_label_templates(features, label_scores, contributions)
        self._apply_contradiction_penalties(features, label_scores, contributions)

        if centralized.predicted_attack_type in label_scores:
            backstop = self.tunable_params["centralized_scale"] * float(centralized.confidence)
            label_scores[centralized.predicted_attack_type] += backstop
            contributions.append(
                FactorContribution(
                    label=centralized.predicted_attack_type,
                    source="centralized_backstop",
                    weight=backstop,
                )
            )

        ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_attack_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        probabilities = self._softmax(np.array([score for _, score in ranked], dtype=float))
        confidence = self._bounded_confidence(float(probabilities[0]))

        trigger_reasons: list[str] = []
        if confidence < self.tunable_params["confidence_trigger_threshold"]:
            trigger_reasons.append("energy_low_confidence")
        if margin < self.tunable_params["margin_trigger_threshold"]:
            trigger_reasons.append("energy_low_margin")
        if predicted_attack_type != "none" and len(label_support.get(predicted_attack_type, [])) < 2:
            trigger_reasons.append("energy_sparse_support")
        if predicted_attack_type == "none" and features["temporal_gap"] >= 0.40:
            trigger_reasons.append("energy_temporal_conflict")
        if predicted_attack_type == "none" and features["event_inconsistency"] >= 0.35:
            trigger_reasons.append("energy_event_conflict")

        top_factors = self._top_factors(contributions, predicted_attack_type)
        enriched_evidence = dict(evidence)
        enriched_evidence["energy_verifier_top_label_score"] = round(float(top_score), 3)
        enriched_evidence["energy_verifier_runner_up_score"] = round(float(second_score), 3)
        enriched_evidence["energy_verifier_margin"] = round(float(margin), 3)
        enriched_evidence["energy_verifier_supporting_agents"] = label_support.get(predicted_attack_type, [])
        enriched_evidence["energy_verifier_top_factors"] = top_factors
        enriched_evidence["energy_verifier_feature_snapshot"] = {
            key: round(float(value), 3) for key, value in features.items()
        }

        verdict = "accept" if predicted_attack_type == "none" else "reject"
        if trigger_reasons:
            verdict = "uncertain"
        return HeuristicAssessment(
            verdict=verdict,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            should_invoke_llm=bool(trigger_reasons),
            trigger_reasons=trigger_reasons,
            evidence=enriched_evidence,
        )

    def _apply_claim_edges(
        self,
        agent_claims: list[AgentClaim],
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
    ) -> None:
        for i, left in enumerate(agent_claims):
            for right in agent_claims[i + 1 :]:
                if left.label == right.label:
                    bonus = 0.10 if left.label != "none" else 0.04
                    bonus *= self.tunable_params["agreement_scale"]
                    label_scores[left.label] += bonus
                    contributions.append(
                        FactorContribution(
                            label=left.label,
                            source=f"agreement:{left.role}<->{right.role}",
                            weight=bonus,
                        )
                    )
                elif left.role == right.role:
                    continue
                else:
                    for conflicting_label in {left.label, right.label} - {"none"}:
                        penalty = -0.05 * self.tunable_params["contradiction_scale"]
                        label_scores[conflicting_label] += penalty
                        contributions.append(
                            FactorContribution(
                                label=conflicting_label,
                                source=f"disagreement:{left.role}<->{right.role}",
                                weight=penalty,
                            )
                        )

        for claim in agent_claims:
            if (
                claim.role == "physical_consistency"
                and claim.label in {"sensor_foil", "gateway_bias"}
                and float(claim.confidence) >= 0.8
            ):
                bonus = 0.22 * self.tunable_params["agreement_scale"]
                label_scores[claim.label] += bonus
                label_scores["random_noise"] -= bonus
                contributions.append(
                    FactorContribution(
                        label=claim.label,
                        source=f"bonus:{claim.role}_directional_claim",
                        weight=bonus,
                    )
                )
                contributions.append(
                    FactorContribution(
                        label="random_noise",
                        source=f"penalty:{claim.role}_directional_claim",
                        weight=-bonus,
                    )
                )

    def _apply_label_templates(
        self,
        features: dict[str, float],
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
    ) -> None:
        for label, weights in self.label_feature_weights.items():
            for feature_name, weight in weights.items():
                contribution = float(weight) * float(features.get(feature_name, 0.0))
                contribution *= self.tunable_params["template_scale"]
                if abs(contribution) < 1e-6:
                    continue
                label_scores[label] += contribution
                contributions.append(
                    FactorContribution(
                        label=label,
                        source=f"feature:{feature_name}",
                        weight=contribution,
                    )
                )

    def _apply_contradiction_penalties(
        self,
        features: dict[str, float],
        label_scores: dict[str, float],
        contributions: list[FactorContribution],
    ) -> None:
        attack_signature = max(
            features["sensor_shift_dominance"],
            features["gateway_shift_dominance"],
            features["variance"],
            features["packet_loss"],
            features["temporal_gap"],
            features["replay_events"],
            features["counter_regressions"],
            features["fabricated_witnesses"],
            features["multiplicity"],
        )
        if attack_signature >= 0.20:
            penalty = -0.75 * attack_signature
            penalty *= self.tunable_params["contradiction_scale"]
            label_scores["none"] += penalty
            contributions.append(
                FactorContribution(
                    label="none",
                    source="penalty:attack_signature_present",
                    weight=penalty,
                )
            )
        if features["environment_plausibility"] >= 0.65:
            for label in ("sensor_foil", "gateway_bias", "random_noise"):
                penalty = -0.20
                penalty *= self.tunable_params["contradiction_scale"]
                label_scores[label] += penalty
                contributions.append(
                    FactorContribution(
                        label=label,
                        source="penalty:high_environment_plausibility",
                        weight=penalty,
                    )
                )
        if features["packet_loss"] >= 0.45 and features["event_inconsistency"] < 0.20:
            bonus = 0.18
            bonus *= self.tunable_params["contradiction_scale"]
            label_scores["packet_drop"] += bonus
            label_scores["selective_suppression"] -= bonus
            contributions.append(
                FactorContribution(
                    label="packet_drop",
                    source="bonus:availability_without_inconsistency",
                    weight=bonus,
                )
            )
            contributions.append(
                FactorContribution(
                    label="selective_suppression",
                    source="penalty:availability_without_inconsistency",
                    weight=-bonus,
                )
            )
        if features["temporal_gap"] >= 0.35 and features["rf_quiet"] >= 0.55:
            bonus = 0.25
            bonus *= self.tunable_params["contradiction_scale"]
            label_scores["replay_attack"] += bonus
            contributions.append(
                FactorContribution(
                    label="replay_attack",
                    source="bonus:quiet_rf_temporal_conflict",
                    weight=bonus,
                )
            )
        if features["delayed_duplicates"] >= 0.10 and features["duplicate_gap"] >= 0.20:
            bonus = 0.45 * max(features["delayed_duplicates"], features["duplicate_gap"])
            bonus *= self.tunable_params["contradiction_scale"]
            label_scores["replay_attack"] += bonus
            label_scores["counter_corruption"] -= bonus
            contributions.append(
                FactorContribution(
                    label="replay_attack",
                    source="bonus:delayed_duplicate_replay",
                    weight=bonus,
                )
            )
            contributions.append(
                FactorContribution(
                    label="counter_corruption",
                    source="penalty:delayed_duplicate_replay",
                    weight=-bonus,
                )
            )
        if features["sensor_shift_dominance"] >= 0.25 and features["variance"] >= 0.70:
            bonus = 0.30 * features["sensor_shift_dominance"]
            bonus *= self.tunable_params["contradiction_scale"]
            label_scores["sensor_foil"] += bonus
            label_scores["random_noise"] -= bonus
            contributions.append(
                FactorContribution(
                    label="sensor_foil",
                    source="bonus:directional_sensor_shift",
                    weight=bonus,
                )
            )
            contributions.append(
                FactorContribution(
                    label="random_noise",
                    source="penalty:directional_sensor_shift",
                    weight=-bonus,
                )
            )
        if features["gateway_shift_dominance"] >= 0.22 and features["variance"] >= 0.45:
            bonus = 0.30 * features["gateway_shift_dominance"]
            bonus *= self.tunable_params["contradiction_scale"]
            label_scores["gateway_bias"] += bonus
            label_scores["random_noise"] -= bonus
            contributions.append(
                FactorContribution(
                    label="gateway_bias",
                    source="bonus:directional_gateway_shift",
                    weight=bonus,
                )
            )
            contributions.append(
                FactorContribution(
                    label="random_noise",
                    source="penalty:directional_gateway_shift",
                    weight=-bonus,
                )
            )
        if features["counter_regressions"] >= 0.40:
            bonus = 0.35 * features["counter_regressions"]
            bonus *= self.tunable_params["contradiction_scale"]
            label_scores["replay_attack"] += bonus
            contributions.append(
                FactorContribution(
                    label="replay_attack",
                    source="bonus:counter_regressions_present",
                    weight=bonus,
                )
            )

    @staticmethod
    def _extract_features(evidence: dict[str, Any]) -> dict[str, float]:
        sensor_shift = float(evidence.get("sensor_mean_abs_rssi_shift_db", 0.0))
        gateway_shift = float(evidence.get("gateway_mean_abs_rssi_shift_db", 0.0))
        sensor_ratio = float(evidence.get("sensor_worst_packet_ratio", 1.0))
        gateway_ratio = float(evidence.get("gateway_worst_packet_ratio", 1.0))
        std_delta = float(evidence.get("sensor_mean_std_delta_db", 0.0))
        residual_delta = float(evidence.get("trilateration_residual_delta_m", 0.0))
        gateway_impacted = int(evidence.get("gateway_impacted_sensor_count", 0))
        max_replay_gap = float(evidence.get("max_replay_gap", 0.0))
        replay_events = int(evidence.get("replay_event_count", 0))
        regressions = int(evidence.get("counter_regression_count", 0))
        delayed_duplicates = int(evidence.get("delayed_duplicate_count", 0))
        duplicate_gap = float(evidence.get("max_duplicate_gap_s", 0.0))
        fabricated = int(evidence.get("fabricated_witness_count", 0))
        multiplicity = int(evidence.get("multiplicity_anomaly_count", 0))
        min_gateway_trust = float(evidence.get("min_gateway_trust", 1.0))
        event_score = float(evidence.get("event_inconsistency_score", 0.0))
        environment_plausibility = max(
            float(evidence.get("sensor_environment_plausibility", 0.0)),
            float(evidence.get("gateway_environment_plausibility", 0.0)),
        )
        top_sensor = float(evidence.get("top_sensor_score", 0.0))
        top_gateway = float(evidence.get("top_gateway_score", 0.0))

        return {
            "rf_quiet": float(np.clip((2.5 - max(top_sensor, top_gateway)) / 2.5, 0.0, 1.0)),
            "environment_plausibility": float(np.clip(environment_plausibility, 0.0, 1.0)),
            "event_inconsistency": float(np.clip(event_score / 4.0, 0.0, 1.0)),
            "low_event_inconsistency": float(np.clip(1.0 - (event_score / 4.0), 0.0, 1.0)),
            "trust_drop": float(np.clip((0.98 - min_gateway_trust) / 0.25, 0.0, 1.0)),
            "low_trust_drop": float(np.clip((min_gateway_trust - 0.75) / 0.23, 0.0, 1.0)),
            "packet_loss": float(np.clip((0.9 - min(sensor_ratio, gateway_ratio)) / 0.9, 0.0, 1.0)),
            "low_packet_loss": float(np.clip(min(sensor_ratio, gateway_ratio), 0.0, 1.0)),
            "sensor_shift_dominance": float(np.clip(max(0.0, sensor_shift - gateway_shift / 2.0) / 12.0, 0.0, 1.0)),
            "gateway_shift_dominance": float(np.clip(max(0.0, gateway_shift - sensor_shift / 2.0) / 12.0, 0.0, 1.0)),
            "variance": float(np.clip(max(std_delta, residual_delta / 10.0) / 4.0, 0.0, 1.0)),
            "gateway_impacted": float(np.clip(gateway_impacted / 5.0, 0.0, 1.0)),
            "temporal_gap": float(np.clip(max_replay_gap / 4.0, 0.0, 1.0)),
            "replay_events": float(np.clip(replay_events / 3.0, 0.0, 1.0)),
            "counter_regressions": float(np.clip(regressions / 100.0, 0.0, 1.0)),
            "delayed_duplicates": float(np.clip(delayed_duplicates / 500.0, 0.0, 1.0)),
            "duplicate_gap": float(np.clip(duplicate_gap / 180.0, 0.0, 1.0)),
            "fabricated_witnesses": float(np.clip(fabricated / 6.0, 0.0, 1.0)),
            "multiplicity": float(np.clip(multiplicity / 4.0, 0.0, 1.0)),
        }

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - float(np.max(values))
        exp_values = np.exp(shifted)
        denom = float(np.sum(exp_values))
        if denom <= 0.0:
            return np.full_like(values, 1.0 / max(values.size, 1), dtype=float)
        return exp_values / denom

    @staticmethod
    def _top_factors(contributions: list[FactorContribution], label: str) -> list[str]:
        top = sorted(
            [item for item in contributions if item.label == label],
            key=lambda item: item.weight,
            reverse=True,
        )[:6]
        return [f"{item.source}:{round(float(item.weight), 3)}" for item in top]

    @staticmethod
    def _bounded_confidence(value: float) -> float:
        return float(np.clip(value, 0.05, 0.99))
