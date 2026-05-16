from dataclasses import dataclass
from typing import Optional

@dataclass
class TransactionPayload:
    transaction_id: str
    tenant_id: str
    amount: float
    currency: str
    merchant_name: str
    raw_description: str
    department: str
    mcc: Optional[str] = None
    auth_vs_settlement: str = "SETTLED" # "AUTH", "SETTLED", "REFUND"

@dataclass
class Categorization:
    account_id: str
    account_name: str
    confidence_score: float
    reasoning: str

@dataclass
class AuditTrail:
    maker_agent: str
    critic_agent: str
    execution_steps: int
    cosine_distance: float
    web_search_invoked: bool = False
    pass_triggered: str = "PASS_2_LLM" # PASS_1_RULES, PASS_2_LLM, PASS_3_CRITIC, DLQ

@dataclass
class AgentOutput:
    transaction_id: str
    tenant_id: str
    tag_status: str # "APPROVED", "NEEDS_REVIEW", "REJECTED", "DLQ"
    categorization: Optional[Categorization] = None
    audit_trail: Optional[AuditTrail] = None
    error_message: Optional[str] = None
