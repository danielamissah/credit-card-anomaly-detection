import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from api.schemas.transaction import TransactionRequest
from api.rules.fraud_rules import (
    evaluate_rules, rule_high_value, rule_foreign_transaction,
    rule_odd_hour, rule_near_credit_limit, rule_atm_foreign, rule_severity_score,
)

BASE = dict(
    transaction_id="T1", user_id=1, amount=50.0, mcc=5411,
    country="DE", home_country="DE", hour=14, day_of_week=1,
    is_card_present=1, credit_limit=5000.0, avg_user_spend=45.0,
)

def txn(**kw): return TransactionRequest(**{**BASE, **kw})

class TestRuleHighValue:
    def test_normal_no_trigger(self):      assert rule_high_value(txn(amount=50)) is None
    def test_10x_triggers(self):           assert rule_high_value(txn(amount=500)) is not None
    def test_20x_is_high(self):            assert rule_high_value(txn(amount=1000)).severity == "high"
    def test_11x_is_medium(self):          assert rule_high_value(txn(amount=550)).severity == "medium"

class TestRuleForeignTransaction:
    def test_domestic_no_trigger(self):    assert rule_foreign_transaction(txn()) is None
    def test_high_risk_is_high(self):      assert rule_foreign_transaction(txn(country="CN")).severity == "high"
    def test_nearby_eu_no_trigger(self):   assert rule_foreign_transaction(txn(country="AT")) is None

class TestRuleOddHour:
    def test_normal_hour_no_trigger(self): assert rule_odd_hour(txn(hour=14, amount=200)) is None
    def test_small_amount_no_trigger(self):assert rule_odd_hour(txn(hour=3, amount=50)) is None
    def test_odd_hour_large_triggers(self):assert rule_odd_hour(txn(hour=3, amount=500)) is not None

class TestRuleNearCreditLimit:
    def test_small_no_trigger(self):       assert rule_near_credit_limit(txn(amount=100)) is None
    def test_above_80pct_triggers(self):   assert rule_near_credit_limit(txn(amount=4100)) is not None

class TestAtmForeign:
    def test_domestic_no_trigger(self):    assert rule_atm_foreign(txn(mcc=6011)) is None
    def test_foreign_triggers(self):       assert rule_atm_foreign(txn(mcc=6011, country="CN")) is not None

class TestEvaluateRules:
    def test_clean_no_rules(self):         assert evaluate_rules(txn()) == []
    def test_suspicious_multiple_rules(self):
        assert len(evaluate_rules(txn(amount=3000, country="CN", mcc=6011, is_card_present=0))) >= 2
    def test_empty_score_zero(self):       assert rule_severity_score([]) == 0.0
    def test_high_rules_score_above_half(self):
        assert rule_severity_score(evaluate_rules(txn(amount=3000, country="CN", mcc=6011))) > 0.5
