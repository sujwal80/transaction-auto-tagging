import pytest
from models.payload import TransactionPayload
from engine.pass1_deterministic import MultiFactorRuleEngine

@pytest.fixture
def rule_engine():
    return MultiFactorRuleEngine()

def test_exact_merchant_match(rule_engine):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=100.0, currency="USD",
        merchant_name="AWS CLOUD COMPUTE", raw_description="AWS", department="IT"
    )
    cat = rule_engine.evaluate(payload)
    assert cat is not None
    assert cat.account_id == "5020"
    assert cat.confidence_score == 1.0

def test_mcc_mapping_match(rule_engine):
    payload = TransactionPayload(
        transaction_id="2", tenant_id="t1", amount=50.0, currency="USD",
        merchant_name="Unknown", raw_description="Food", department="Sales",
        mcc="5734"
    )
    cat = rule_engine.evaluate(payload)
    assert cat is not None
    assert cat.account_id == "5020"

def test_no_deterministic_match(rule_engine):
    payload = TransactionPayload(
        transaction_id="3", tenant_id="t1", amount=50.0, currency="USD",
        merchant_name="Random Store", raw_description="Stuff", department="HR"
    )
    cat = rule_engine.evaluate(payload)
    assert cat is None
