import pytest
from models.payload import TransactionPayload, Categorization
from engine.pass3_guardrails import GuardrailPipeline
import config

@pytest.fixture
def pipeline():
    return GuardrailPipeline()

def test_materiality_spend_cap_filter(pipeline):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=config.MATERIALITY_CAP + 100, currency="USD",
        merchant_name="Test", raw_description="Test", department="Test"
    )
    cat = Categorization("5020", "Software", 0.95, "Test")
    status, reason = pipeline.evaluate(payload, cat, execution_steps=1, cosine_distance=0.1)
    assert status == "NEEDS_REVIEW"
    assert "Materiality Spend Cap exceeded" in reason

def test_vector_distance_filter(pipeline):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=100, currency="USD",
        merchant_name="Test", raw_description="Test", department="Test"
    )
    cat = Categorization("5020", "Software", 0.95, "Test")
    status, reason = pipeline.evaluate(payload, cat, execution_steps=1, cosine_distance=0.40)
    assert status == "NEEDS_REVIEW"
    assert "Out-of-Distribution" in reason

def test_reasoning_depth_filter(pipeline):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=100, currency="USD",
        merchant_name="Test", raw_description="Test", department="Test"
    )
    cat = Categorization("5020", "Software", 0.95, "Test")
    status, reason = pipeline.evaluate(payload, cat, execution_steps=5, cosine_distance=0.1)
    assert status == "NEEDS_REVIEW"
    assert "Reasoning depth anomaly" in reason

def test_confidence_bounding_filter(pipeline):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=100, currency="USD",
        merchant_name="Test", raw_description="Test", department="Test"
    )
    cat = Categorization("5020", "Software", 0.80, "Test")
    status, reason = pipeline.evaluate(payload, cat, execution_steps=1, cosine_distance=0.1)
    assert status == "NEEDS_REVIEW"
    assert "Maker confidence score insufficient" in reason

def test_dual_agent_critic_objection(pipeline):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=1500, currency="USD",
        merchant_name="UBER", raw_description="Test", department="Test"
    )
    cat = Categorization("5030", "Travel", 0.95, "Test")
    status, reason = pipeline.evaluate(payload, cat, execution_steps=1, cosine_distance=0.1)
    assert status == "NEEDS_REVIEW"
    assert "Critic Agent Objection" in reason

def test_approved_path(pipeline):
    payload = TransactionPayload(
        transaction_id="1", tenant_id="t1", amount=50, currency="USD",
        merchant_name="AWS", raw_description="Test", department="Test"
    )
    cat = Categorization("5020", "Software", 0.95, "Test")
    status, reason = pipeline.evaluate(payload, cat, execution_steps=1, cosine_distance=0.1)
    assert status == "APPROVED"
    assert reason is None
