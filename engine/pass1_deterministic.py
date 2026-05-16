from typing import Optional
from models.payload import TransactionPayload, Categorization

class MultiFactorRuleEngine:
    """
    High-speed, zero-latency Pass 1 filter.
    Evaluates transactions against strict, exact-match and regex mapping matrices.
    Bypasses AI entirely if a deterministic match is found.
    """
    def __init__(self):
        # In a real system, this would be loaded from a database per tenant
        self.exact_merchant_rules = {
            "AWS CLOUD COMPUTE": ("5020", "Software Subscriptions", "Deterministic rule: Exact merchant AWS"),
            "WEWORK MONTHLY RENT": ("5010", "Hardware & Office Equipment", "Deterministic rule: Exact merchant WeWork"),
        }
        
        self.mcc_mapping = {
            "5734": ("5020", "Software Subscriptions", "Deterministic rule: MCC 5734 (Computer Software)"),
            "3000": ("5030", "Travel & Entertainment", "Deterministic rule: MCC 3000 (Airlines)"),
        }

    def evaluate(self, payload: TransactionPayload) -> Optional[Categorization]:
        # 1. Evaluate MCC (Virtual Card Deterministic Grounding)
        if payload.mcc and payload.mcc in self.mcc_mapping:
            acc_id, acc_name, reason = self.mcc_mapping[payload.mcc]
            return Categorization(
                account_id=acc_id,
                account_name=acc_name,
                confidence_score=1.0,
                reasoning=reason
            )
            
        # 2. Evaluate Exact Merchant Match (No Fuzzy Matching in Pass 1)
        if payload.merchant_name in self.exact_merchant_rules:
            acc_id, acc_name, reason = self.exact_merchant_rules[payload.merchant_name]
            return Categorization(
                account_id=acc_id,
                account_name=acc_name,
                confidence_score=1.0,
                reasoning=reason
            )
            
        # If no deterministic rule matches, return None to trigger Pass 2
        return None
