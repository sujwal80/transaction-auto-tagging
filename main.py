import asyncio
import json
import dataclasses
from typing import Any
from models.payload import TransactionPayload, AgentOutput
from worker.daemon import TransactionWorkerDaemon

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

async def run_simulation():
    daemon = TransactionWorkerDaemon()
    
    # ========================================================================
    # SCENARIO 1: Pass 1 Deterministic Rule Match
    # Expected: APPROVED via PASS_1_RULES
    # ========================================================================
    tx1 = TransactionPayload(
        transaction_id="tx_001",
        tenant_id="tenant_acme_01",
        amount=1250.00,
        currency="USD",
        merchant_name="AWS CLOUD COMPUTE",
        raw_description="INV-2026-05 AWS CLOUD COMPUTE US-EAST-1",
        department="Engineering",
        mcc="5734"
    )
    await daemon.publish_event(tx1)
    
    # ========================================================================
    # SCENARIO 2: Pass 2 Maker LLM Match (Standard RAG)
    # Expected: APPROVED via PASS_2_LLM
    # ========================================================================
    tx2 = TransactionPayload(
        transaction_id="tx_002",
        tenant_id="tenant_acme_01",
        amount=240.00,
        currency="USD",
        merchant_name="FACEBOOK ADS",
        raw_description="FACEBOOK ADS CAMPAIGN MAY 2026", # Matches substring
        department="Marketing",
        mcc=None
    )
    await daemon.publish_event(tx2)
    
    # ========================================================================
    # SCENARIO 3: Pass 3 Materiality Spend Cap Exceeded ($5,000+)
    # Expected: NEEDS_REVIEW (Early Exit)
    # ========================================================================
    tx3 = TransactionPayload(
        transaction_id="tx_003",
        tenant_id="tenant_acme_01",
        amount=12000.00,
        currency="USD",
        merchant_name="SKADDEN ARPS RETAINER",
        raw_description="SKADDEN ARPS RETAINER 02",
        department="Legal",
        mcc=None
    )
    await daemon.publish_event(tx3)
    
    # ========================================================================
    # SCENARIO 4: Pass 3 Audit Critic Objection (Policy Violation)
    # Expected: NEEDS_REVIEW (Critic Objection)
    # ========================================================================
    tx4 = TransactionPayload(
        transaction_id="tx_004",
        tenant_id="tenant_acme_01",
        amount=1500.00, # Exceeds $1000 travel policy
        currency="USD",
        merchant_name="UBER RIDE",
        raw_description="UBER RIDE LUXURY CHARTER", # Matches substring for distance 0.10
        department="Sales",
        mcc=None
    )
    await daemon.publish_event(tx4)
    
    # ========================================================================
    # SCENARIO 5: Poison Pill Intercept (Malformed JSON)
    # Expected: DLQ Routing without crashing worker
    # ========================================================================
    poison_pill = {"corrupted_id": "null", "broken_json": True}
    await daemon.publish_poison_pill(poison_pill)
    
    # Process all 5 items in queue
    print("\n" + "="*80)
    print("STARTING ASYNC WORKER DAEMON SIMULATION (ZERO-TOLERANCE PIPELINE)")
    print("="*80 + "\n")
    
    results = []
    for i in range(5):
        output = await daemon.process_queue_step()
        results.append(output)
        
    print("\n" + "="*80)
    print("SIMULATION COMPLETED. VERIFYING OUTPUT PAYLOADS:")
    print("="*80 + "\n")
    
    for res in results:
        print(json.dumps(res, cls=EnhancedJSONEncoder, indent=2))
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(run_simulation())
