import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.training.embed_utils import encode_cached, cosine_sim
from src.training.grounded_rules import (
    ALL_TOPIC_RULES,
    ALL_ACTION_RULES,
)

@dataclass
class Predicate:
    name: str
    patterns: list[str]
    source: str = ""
    anchor_text: str = ""

    def evaluate(self, text: str) -> bool:
        text_lower = text.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def fuzzy_evaluate(
        self,
        text: str,
        encoder,
    ) -> float:
        """
        match_count = hard regex matches + 0.5 x semantic_sim(text, anchor)
        t = min(match_count × 0.5, 1.0)
        """
        text_lower = text.lower()
        hard_count = sum(
            1 for p in self.patterns
            if re.search(p, text_lower, re.IGNORECASE)
        )

        sem_contribution = 0.0
        anchor = self.anchor_text or self.source
        if anchor:
            text_emb   = encode_cached(encoder, text)
            anchor_emb = encode_cached(encoder, anchor)
            sim = cosine_sim(text_emb, anchor_emb)
            sem_contribution = 0.5 * max(0.0, sim)

        soft_count = hard_count + sem_contribution
        return min(soft_count * 0.5, 1.0)

@dataclass
class Constraint:
    name: str
    constraint_type: str
    predicate: Optional[Predicate] = None
    target_label_idx: Optional[int] = None
    source: str = ""

    def __post_init__(self):
        if self.constraint_type not in ("implication", "negation", "exactly_one", "mutual_exclusion"):
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")

class PropositionalKnowledgeBase:
    def __init__(self, labels: list[str], task: str = "topic", encoder=None):
        self.labels = labels
        self.task = task
        self.encoder = encoder
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.constraints: list[Constraint] = []
        self.predicates: dict[str, list[Predicate]] = {}

        self._build_knowledge_base()

    def _build_knowledge_base(self):
        self.constraints.append(Constraint(
            name="exactly_one",
            constraint_type="exactly_one",
            source="Classification axiom: single-label constraint",
        ))

        rule_source = ALL_TOPIC_RULES if self.task == "topic" else ALL_ACTION_RULES

        for label, rules in rule_source.items():
            if label not in self.label_to_idx:
                continue

            label_idx = self.label_to_idx[label]
            label_predicates = []

            for rule in rules:
                pred = Predicate(
                    name=rule.name,
                    patterns=rule.patterns,
                    source=rule.source,
                    anchor_text=getattr(rule, "anchor_text", ""),
                )
                label_predicates.append(pred)

                self.constraints.append(Constraint(
                    name=f"impl_{rule.name}_{label}",
                    constraint_type="implication",
                    predicate=pred,
                    target_label_idx=label_idx,
                    source=rule.source,
                ))

            self.predicates[label] = label_predicates

        if self.task == "action":
            self._build_action_negation_constraints()

    def _build_action_negation_constraints(self):
        if "Implemented" not in self.label_to_idx:
            return

        impl_idx = self.label_to_idx["Implemented"]

        for rule in ALL_ACTION_RULES.get("Indeterminate", []):
            pred = Predicate(
                name=rule.name,
                patterns=rule.patterns,
                source=rule.source,
                anchor_text=getattr(rule, "anchor_text", ""),
            )
            self.constraints.append(Constraint(
                name=f"neg_{rule.name}_impl",
                constraint_type="negation",
                predicate=pred,
                target_label_idx=impl_idx,
                source=f"{rule.source} → ¬Implemented",
            ))

class SemanticLoss(nn.Module):
    def __init__(
        self,
        knowledge_base: PropositionalKnowledgeBase,
        lambda_weight: float = 0.3,
        exactly_one_weight: float = 1.0,
        implication_weight: float = 0.5,
        negation_weight: float = 0.5,
        eps: float = 1e-8,
        encoder=None,
    ):
        super().__init__()
        self.kb = knowledge_base
        if encoder is not None:
            self.kb.encoder = encoder
        self.lambda_weight = lambda_weight
        self.exactly_one_weight = exactly_one_weight
        self.implication_weight = implication_weight
        self.negation_weight = negation_weight
        self.eps = eps

    def exactly_one_loss(self, probs: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        one_minus_p = 1.0 - probs

        prod_all = torch.prod(one_minus_p, dim=1)

        ratio_sum = torch.sum(probs / one_minus_p, dim=1)

        wmc = prod_all * ratio_sum

        loss = -torch.log(wmc + self.eps)

        return loss.mean()

    def _truth(self, predicate: "Predicate", text: str) -> float:
        return predicate.fuzzy_evaluate(text, self.kb.encoder)

    def implication_loss(
        self,
        probs: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:
        device = probs.device
        total_loss = torch.tensor(0.0, device=device)
        active_count = 0

        for i, text in enumerate(texts):
            for constraint in self.kb.constraints:
                if constraint.constraint_type != "implication":
                    continue
                if constraint.predicate is None or constraint.target_label_idx is None:
                    continue

                truth = self._truth(constraint.predicate, text)
                if truth <= 0:
                    continue

                target_prob = probs[i, constraint.target_label_idx]
                target_prob = torch.clamp(target_prob, min=self.eps)
                total_loss = total_loss + (-truth * torch.log(target_prob))
                active_count += 1

        if active_count > 0:
            return total_loss / active_count
        return total_loss

    def negation_loss(
        self,
        probs: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:
        device = probs.device
        total_loss = torch.tensor(0.0, device=device)
        active_count = 0

        for i, text in enumerate(texts):
            for constraint in self.kb.constraints:
                if constraint.constraint_type != "negation":
                    continue
                if constraint.predicate is None or constraint.target_label_idx is None:
                    continue

                truth = self._truth(constraint.predicate, text)
                if truth <= 0:
                    continue

                target_prob = probs[i, constraint.target_label_idx]
                one_minus_p = torch.clamp(1.0 - target_prob, min=self.eps)
                total_loss = total_loss + (-truth * torch.log(one_minus_p))
                active_count += 1

        if active_count > 0:
            return total_loss / active_count
        return total_loss

    def forward(
        self,
        logits: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)

        loss_eo = self.exactly_one_weight * self.exactly_one_loss(probs)

        loss_impl = self.implication_weight * self.implication_loss(probs, texts)

        loss_neg = self.negation_weight * self.negation_loss(probs, texts)

        total = loss_eo + loss_impl + loss_neg

        return self.lambda_weight * total

_TOPIC_LABELS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]
_ACTION_LABELS = ["Implemented", "Planning", "Indeterminate"]

def _resolve_task_labels(task: str, labels: Optional[list[str]]) -> list[str]:
    if labels:
        return labels
    return _TOPIC_LABELS if task == "topic" else _ACTION_LABELS

def _resolve_neuro_symbolic_config(config: Optional[dict]) -> dict:
    if not isinstance(config, dict):
        return {}

    if isinstance(config.get("neuro_symbolic"), dict):
        return config.get("neuro_symbolic", {})

    return {}

def create_semantic_loss(
    task: str = "topic",
    labels: Optional[list[str]] = None,
    config: Optional[dict] = None,
    encoder=None,
) -> SemanticLoss:
    ns_cfg = _resolve_neuro_symbolic_config(config)
    resolved_labels = _resolve_task_labels(task, labels=labels)

    kb = PropositionalKnowledgeBase(resolved_labels, task, encoder=encoder)

    return SemanticLoss(
        knowledge_base=kb,
        lambda_weight=float(ns_cfg.get("constraint_lambda", 0.3)),
        exactly_one_weight=float(ns_cfg.get("exactly_one_weight", 1.0)),
        implication_weight=float(ns_cfg.get("implication_weight", 0.5)),
        negation_weight=float(ns_cfg.get("negation_weight", 0.5)),
    )
