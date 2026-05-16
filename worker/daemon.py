import asyncio
import logging
from typing import Optional, Dict, Any
from models.payload import TransactionPayload, AgentOutput, AuditTrail, Categorization
from models.coa import ChartOfAccountsStore
from engine.pass1_deterministic import MultiFactorRuleEngine
from engine.pass2_maker_llm import MakerAgent
from engine.pass3_guardrails import GuardrailPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WorkerDaemon")

class TransactionWorkerDaemon:
    """
    Emulates the asynchronous RabbitMQ worker daemon pulling from raw event queues.
    Executes the Pass 1 -> Pass 2 -> Pass 3 pipeline.
    Implements strict ACK on success and routes unparseable / failing messages to a DLQ.
    """
    def __init__(self):
        self.raw_event_queue = asyncio.Queue()
        self.dlq = asyncio.Queue()
        self.processed_results = asyncio.Queue()
        
        self.coa_store = ChartOfAccountsStore()
        self.rule_engine = MultiFactorRuleEngine()
        self.maker_agent = MakerAgent(self.coa_store)
        self.guardrails = GuardrailPipeline()
        
    async def publish_event(self, payload: TransactionPayload):
        """Emulates an upstream service publishing to RabbitMQ."""
        await self.raw_event_queue.put(payload)
        logger.info(f"Published event {payload.transaction_id} to raw queue.")
        
    async def publish_poison_pill(self, malformed_data: Dict[str, Any]):
        """Emulates an unparseable JSON payload arriving in the queue."""
        await self.raw_event_queue.put(malformed_data)
        logger.info("Published malformed poison pill to raw queue.")
        
    async def process_queue_step(self) -> AgentOutput:
        """
        Pulls one item from queue and processes it.
        Returns AgentOutput for verification.
        """
        raw_item = await self.raw_event_queue.get()
        
        # Check if poison pill (not a TransactionPayload)
        if not isinstance(raw_item, TransactionPayload):
            logger.warning(f"Malformed payload detected! Routing to Dead Letter Queue (DLQ).")
            await self.dlq.put(raw_item)
            self.raw_event_queue.task_done()
            return AgentOutput(
                transaction_id="unknown",
                tenant_id="unknown",
                tag_status="DLQ",
                error_message="Payload parsing failed (Poison Pill intercepted)."
            )
            
        payload: TransactionPayload = raw_item
        logger.info(f"Processing transaction {payload.transaction_id} (${payload.amount:.2f})...")
        
        try:
            # ==========================================================
            # PASS 1: Multi-Factor Deterministic Rule Engine
            # ==========================================================
            cat = self.rule_engine.evaluate(payload)
            if cat:
                logger.info(f"[{payload.transaction_id}] Pass 1 Deterministic Match: {cat.account_name}")
                output = AgentOutput(
                    transaction_id=payload.transaction_id,
                    tenant_id=payload.tenant_id,
                    tag_status="APPROVED",
                    categorization=cat,
                    audit_trail=AuditTrail(
                        maker_agent="DeterministicRuleEngine",
                        critic_agent="None",
                        execution_steps=0,
                        cosine_distance=0.0,
                        pass_triggered="PASS_1_RULES"
                    )
                )
                await self.processed_results.put(output)
                self.raw_event_queue.task_done()
                return output
                
            # ==========================================================
            # PASS 2: Multi-Tenant RAG & LLM Maker Agent
            # ==========================================================
            logger.info(f"[{payload.transaction_id}] Pass 1 missed. Delegating to Pass 2 Maker Agent...")
            cat, steps, distance = await self.maker_agent.generate_categorization(payload)
            
            # ==========================================================
            # PASS 3: Lean MVP Guardrail Pipeline & Dual-Agent Critic
            # ==========================================================
            logger.info(f"[{payload.transaction_id}] Executing Pass 3 Guardrail Verification...")
            tag_status, rejection_reason = self.guardrails.evaluate(payload, cat, steps, distance)
            
            audit = AuditTrail(
                maker_agent="Gemini Flash 1.5",
                critic_agent="Claude Opus",
                execution_steps=steps,
                cosine_distance=distance,
                pass_triggered="PASS_3_CRITIC" if tag_status == "NEEDS_REVIEW" else "PASS_2_LLM"
            )
            
            if tag_status == "NEEDS_REVIEW":
                logger.warning(f"[{payload.transaction_id}] Early Exit Triggered ({tag_status}): {rejection_reason}")
                output = AgentOutput(
                    transaction_id=payload.transaction_id,
                    tenant_id=payload.tenant_id,
                    tag_status=tag_status,
                    categorization=cat, # Keep category for human review suggestion
                    audit_trail=audit,
                    error_message=rejection_reason
                )
            else:
                logger.info(f"[{payload.transaction_id}] Approved by Dual-Agent Critic: {cat.account_name}")
                output = AgentOutput(
                    transaction_id=payload.transaction_id,
                    tenant_id=payload.tenant_id,
                    tag_status=tag_status,
                    categorization=cat,
                    audit_trail=audit
                )
                
            await self.processed_results.put(output)
            self.raw_event_queue.task_done() # ACK message
            return output
            
        except Exception as e:
            # Catch unexpected worker crashes (e.g. API outage) and route to DLQ
            logger.error(f"[{payload.transaction_id}] Unhandled worker exception: {str(e)}. Routing to DLQ.")
            await self.dlq.put(payload)
            self.raw_event_queue.task_done()
            return AgentOutput(
                transaction_id=payload.transaction_id,
                tenant_id=payload.tenant_id,
                tag_status="DLQ",
                error_message=f"Unhandled exception: {str(e)}"
            )
