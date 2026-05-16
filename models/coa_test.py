import pytest
from models.coa import ChartOfAccountsStore

@pytest.fixture
def store():
    return ChartOfAccountsStore()

def test_get_coa_mapping(store):
    mapping = store.get_coa_mapping("tenant_acme_01")
    assert "5020" in mapping
    assert mapping["5020"] == "Software Subscriptions"

def test_query_vector_index_exact(store):
    acc, dist = store.query_vector_index("tenant_acme_01", "UBER RIDE")
    assert acc == "5030"
    assert dist == 0.05

def test_query_vector_index_fuzzy(store):
    acc, dist = store.query_vector_index("tenant_acme_01", "SOME FACEBOOK ADS CAMPAIGN")
    assert acc == "5050"
    assert dist == 0.11

def test_query_vector_index_ood(store):
    acc, dist = store.query_vector_index("tenant_acme_01", "RANDOM STRING NO MATCH")
    assert acc == "5020"
    assert dist == 0.35
