import pytest
from models.payload import TransactionPayload
from models.coa import ChartOfAccountsStore
from engine.pass2_maker_llm import MakerAgent
import config

@pytest.fixture
def maker_agent():
    store = ChartOfAccountsStore()
    return MakerAgent(store)

@pytest.mark.asyncio
async def test_generate_categorization_mock_facebook(maker_agent, monkeypatch):
    monkeypatch.setattr(config, "SIMULATION_MODE", True)
    payload = TransactionPayload(
        transaction_id="1", tenant_id="tenant_acme_01", amount=100.0, currency="USD",
        merchant_name="FACEBOOK ADS", raw_description="FACEBOOK ADS CAMPAIGN", department="Marketing"
    )
    cat, steps, distance = await maker_agent.generate_categorization(payload)
    assert cat is not None
    assert cat.account_id == "5050"
    assert steps == 1
    assert distance == 0.11
    assert cat.confidence_score == pytest.approx(0.835, abs=0.01)

@pytest.mark.asyncio
async def test_generate_categorization_high_reasoning_steps(maker_agent, monkeypatch):
    monkeypatch.setattr(config, "SIMULATION_MODE", True)
    payload = TransactionPayload(
        transaction_id="2", tenant_id="tenant_acme_01", amount=100.0, currency="USD",
        merchant_name="CRYPTO EXCHANGE", raw_description="BUY BITCOIN CRYPTO", department="Finance"
    )
    cat, steps, distance = await maker_agent.generate_categorization(payload)
    assert steps == 3
    assert distance == 0.45
