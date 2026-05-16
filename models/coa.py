from typing import Dict, Tuple

class ChartOfAccountsStore:
    """
    Manages tenant-specific Chart of Accounts definitions and emulates
    a read-only vector index for RAG context assembly.
    """
    def __init__(self):
        self.coa_definitions = {
            "tenant_acme_01": {
                "5010": "Hardware & Office Equipment",
                "5020": "Software Subscriptions",
                "5030": "Travel & Entertainment",
                "5040": "Legal & Professional Fees",
                "5050": "Marketing & Advertising",
            }
        }
        
        # Emulated vector embeddings for known phrases
        # In a real system, this would be a vector DB lookup
        self.mock_vector_index = {
            "tenant_acme_01": {
                "INV-2026-05 GITHUB ENTERPRISE": ("5020", 0.14), # (account_id, cosine_distance)
                "AWS CLOUD COMPUTE": ("5020", 0.12),
                "UBER RIDE": ("5030", 0.05), # Low distance so confidence > 0.85 to hit Critic
                "WEWORK MONTHLY RENT": ("5010", 0.22),
                "SKADDEN ARPS RETAINER": ("5040", 0.15),
                "FACEBOOK ADS": ("5050", 0.11),
                # OOD example
                "BUY BITCOIN CRYPTO": ("5050", 0.45), # high distance
            }
        }
        
    def get_coa_mapping(self, tenant_id: str) -> Dict[str, str]:
        return self.coa_definitions.get(tenant_id, {})
        
    def query_vector_index(self, tenant_id: str, query_text: str) -> Tuple[str, float]:
        """
        Emulates a vector search returning (closest_account_id, cosine_distance).
        """
        tenant_index = self.mock_vector_index.get(tenant_id, {})
        
        # Exact mock match
        if query_text in tenant_index:
            return tenant_index[query_text]
            
        # Fuzzy mock fallback
        for key, val in tenant_index.items():
            if key in query_text or query_text in key:
                return val
                
        # Cold start / OOD fallback
        return ("5020", 0.35) # returns high distance above threshold 0.30
