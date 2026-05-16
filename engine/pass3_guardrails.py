from typing import Optional, Tuple
from models.payload import TransactionPayload, Categorization, AuditTrail
import config

class AuditCriticAgent:
    """
    Emulates Claude Opus / larger auditing LLM.
    Evaluates Maker output against negative CoA constraints.
    """
    def evaluate_policy(self, payload: TransactionPayload, cat: Categorization) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_approved, objection_reason).
        """
        # Negative Policy Constraint 1: Travel spend > $1000 requires manual review
        if cat.account_id == "5030" and payload.amount > 1000.0:
            return False, "Policy violation: Travel & Entertainment spend over $1,000 requires expense receipt audit."
            
        # Negative Policy Constraint 2: Marketing spend over $5000 requires VP approval
        if cat.account_id == "5040" and payload.amount > 5000.0:
            return False, "Policy violation: Marketing spend over $5,000 requires executive sign-off."
            
        return True, None

class GuardrailPipeline:
    """
    Validates materiality spend caps, vector distance anomalies, and reasoning depth bounds.
    """
    def __init__(self):
        self.critic = AuditCriticAgent()
        
    def evaluate(
        self,
        payload: TransactionPayload,
        cat: Categorization,
        execution_steps: int,
        cosine_distance: float
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (tag_status, rejection_reason).
        """
        # 1. Materiality Spend Cap Filter
        if payload.amount >= config.MATERIALITY_CAP:
            return "NEEDS_REVIEW", f"Materiality Spend Cap exceeded (${payload.amount:,.2f} >= ${config.MATERIALITY_CAP:,.2f})."
            
        # 2. Vector Distance Bounding Filter
        if cosine_distance > config.VECTOR_DISTANCE_THRESHOLD:
            return "NEEDS_REVIEW", f"Out-of-Distribution anomaly detected (cosine distance {cosine_distance} > {config.VECTOR_DISTANCE_THRESHOLD})."
            
        # 3. Reasoning Depth Bounding Filter
        if execution_steps > config.MAX_REASONING_STEPS:
            return "NEEDS_REVIEW", f"Reasoning depth anomaly detected ({execution_steps} steps > baseline max {config.MAX_REASONING_STEPS})."
            
        # 4. LLM Confidence Bounding Filter
        if cat.confidence_score < config.CONFIDENCE_THRESHOLD:
            return "NEEDS_REVIEW", f"Maker confidence score insufficient ({cat.confidence_score} < {config.CONFIDENCE_THRESHOLD})."
            
        # 5. Dual-Agent Audit Critic Policy Evaluation
        is_approved, objection = self.critic.evaluate_policy(payload, cat)
        if not is_approved:
            return "NEEDS_REVIEW", f"Critic Agent Objection: {objection}"
            
        return "APPROVED", None
