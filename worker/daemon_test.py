import pytest
import asyncio
from models.payload import TransactionPayload
from worker.daemon import TransactionWorkerDaemon

@pytest.fixture
def daemon():
    return TransactionWorkerDaemon()

@pytest.mark.asyncio
async def test_process_queue_poison_pill(daemon):
    poison_pill = {"corrupted": True}
    await daemon.publish_poison_pill(poison_pill)
    output = await daemon.process_queue_step()
    assert output.tag_status == "DLQ"
    assert output.transaction_id == "unknown"
    assert "Payload parsing failed" in output.error_message

@pytest.mark.asyncio
async def test_process_queue_deterministic_match(daemon):
    tx = TransactionPayload(
        transaction_id="tx_test", tenant_id="t1", amount=100.0, currency="USD",
        merchant_name="AWS CLOUD COMPUTE", raw_description="AWS", department="IT", mcc="5734"
    )
    await daemon.publish_event(tx)
    output = await daemon.process_queue_step()
    assert output.tag_status == "APPROVED"
    assert output.categorization.account_id == "5020"
    assert output.audit_trail.pass_triggered == "PASS_1_RULES"
