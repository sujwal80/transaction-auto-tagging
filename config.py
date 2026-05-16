import os

# ==============================================================================
# CONFIGURATION & SHARED SETTINGS
# ==============================================================================

# Materiality Spend Cap threshold ($5,000.00) triggering immediate manual review
MATERIALITY_CAP = 5000.00

# ==============================================================================
# ARCHITECTURAL NOTE (MVP SIMULATION SHORTCUT):
# For this MVP assessment, we utilize a static cosine distance threshold (0.30)
# to demonstrate the early-exit refusal pipeline (NEEDS_REVIEW).
# 
# In production, static thresholds are brittle across different embedding
# models. The target architecture replaces this with Nearest Neighbor Distance
# Ratios (D1/D2) and Tenant-Specific Z-Scores for dynamic anomaly bounding.
# ==============================================================================
VECTOR_DISTANCE_THRESHOLD = 0.30

# Confidence threshold for LLM Maker output acceptance
CONFIDENCE_THRESHOLD = 0.85

# Maximum ReAct reasoning loops before triggering anomaly early exit
MAX_REASONING_STEPS = 2

# Simulation Mode: True for deterministic offline mocking, False for live SDK
SIMULATION_MODE = os.getenv("LIVE_GENAI_API_KEY") is None
