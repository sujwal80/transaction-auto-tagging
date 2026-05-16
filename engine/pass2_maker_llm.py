import os
import json
import logging
from typing import Tuple, Optional
from models.payload import TransactionPayload, Categorization
from models.coa import ChartOfAccountsStore
import config

logger = logging.getLogger("MakerAgent")

class MakerAgent:
    """
    Orchestrates Pass 2 RAG Context Assembly and LLM categorization
    generation.
    Supports Mock Adapter Mode and Live SDK Mode.
    """
    def __init__(self, coa_store: ChartOfAccountsStore):
        self.coa_store = coa_store
        self.api_key = os.getenv("GEMINI_API_KEY")

    async def generate_categorization(
        self, payload: TransactionPayload
    ) -> Tuple[Optional[Categorization], int, float]:
        """
        Returns (Categorization, execution_steps, cosine_distance).
        """
        tenant_coa = self.coa_store.get_coa_mapping(payload.tenant_id)

        # Simulate RAG lookup
        closest_acc, cosine_distance = \
        self.coa_store.query_vector_index(
            payload.tenant_id, payload.raw_description
        )

        # Simulate ReAct loop counter
        execution_steps = 1
        if "CRYPTO" in payload.raw_description or "BITCOIN" in payload.raw_description:
            execution_steps = 3  # Exceeds MAX_REASONING_STEPS (2)

        # If simulation mode or no API key, use mock generation
        if config.SIMULATION_MODE or not self.api_key:
            return self.mock_llm_generation(
                payload, tenant_coa, closest_acc, cosine_distance,
                execution_steps
            )
        else:
            return await self.live_gemini_generation(
                payload, tenant_coa, closest_acc, cosine_distance,
                execution_steps
            )

    async def _live_gemini_generation(
        self,
        payload: TransactionPayload,
        coa_map: dict,
        closest_acc: str,
        cosine_distance: float,
        execution_steps: int
    ) -> Tuple[Optional[Categorization], int, float]:
        """
        Connects to live Google GenAI API using Structured Outputs.
        Falls back to mock if API fails.
        """
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)

            # Define strict JSON schema for structured output
            output_schema = {
                "type": "OBJECT",
                "properties": {
                    "account_id": {"type": "STRING"},
                    "account_name": {"type": "STRING"},
                    "confidence_score": {"type": "NUMBER"},
                    "reasoning": {"type": "STRING"}
                },
                "required": ["account_id", "account_name", "confidence_score", "reasoning"]
            }

            prompt = f"""You are an expert financial tagging AI with zero tolerance for errors.
Categorize the following transaction into the provided Chart of Accounts.

Transaction: {payload.raw_description} (Amount: ${payload.amount})
Merchant: {payload.merchant_name}

Chart of Accounts:
{json.dumps(coa_map, indent=2)}

Provide the account_id, account_name, your confidence score (0.0 to 1.0), and detailed reasoning.
"""

            logger.info(f"Calling live Gemini Flash 1.5 API for tx {payload.transaction_id}...")
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=output_schema,
                    temperature=0.1
                )
            )

            data = json.loads(response.text)
            cat = Categorization(
                account_id=data["account_id"],
                account_name=data["account_name"],
                confidence_score=float(data["confidence_score"]),
                reasoning=data["reasoning"]
            )
            return cat, execution_steps, cosine_distance

        except Exception as e:
            logger.warning(f"Live Gemini API failed: {str(e)}. Falling back to deterministic mock.")
            return self.mock_llm_generation(
                payload, coa_map, closest_acc, cosine_distance,
                execution_steps
            )

    def mock_llm_generation(
        self,
        payload: TransactionPayload,
        coa_map: dict,
        closest_acc: str,
        cosine_distance: float,
        execution_steps: int
    ) -> Tuple[Optional[Categorization], int, float]:
        
        account_name = coa_map.get(closest_acc, "Unknown Account")
        confidence = max(0.10, 1.0 - (cosine_distance * 1.5))

        cat = Categorization(
            account_id=closest_acc,
            account_name=account_name,
            confidence_score=round(confidence, 2),
            reasoning=f"Maker Gemini Flash 1.5 matched based on RAG semantic similarity to CoA {closest_acc}."
        )

        return cat, execution_steps, cosine_distance