from typing import Optional
from api.schemas.transaction import TransactionRequest, RuleTrigger

HIGH_RISK_COUNTRIES = {"NG","CN","RU","UA","BR","VN","PK","ID","RO","BG"}
LOW_RISK_COUNTRIES  = {"DE","AT","CH","FR","NL","BE","LU","DK","SE","NO","FI","GB","IE","ES","PT","IT","PL","CZ","HU","SK"}
HIGH_RISK_MCC       = {6011, 6051, 7801, 7995, 5944, 5912}


def rule_high_value(txn: TransactionRequest) -> Optional[RuleTrigger]:
    ratio = txn.amount / (txn.avg_user_spend + 1e-6)
    if ratio >= 10.0:
        return RuleTrigger(
            rule_name   = "high_value_ratio",
            description = f"Amount EUR{txn.amount:.2f} is {ratio:.1f}x user avg (EUR{txn.avg_user_spend:.2f})",
            severity    = "high" if ratio >= 20 else "medium",
            value       = round(ratio, 2),
            threshold   = 10.0,
        )
    return None


def rule_near_credit_limit(txn: TransactionRequest) -> Optional[RuleTrigger]:
    ratio = txn.amount / (txn.credit_limit + 1e-6)
    if ratio >= 0.80:
        return RuleTrigger(
            rule_name   = "near_credit_limit",
            description = f"Transaction is {ratio:.0%} of credit limit",
            severity    = "high",
            value       = round(ratio, 4),
            threshold   = 0.80,
        )
    return None


def rule_foreign_transaction(txn: TransactionRequest) -> Optional[RuleTrigger]:
    if txn.country == txn.home_country:
        return None
    is_high_risk = txn.country in HIGH_RISK_COUNTRIES
    if is_high_risk or txn.country not in LOW_RISK_COUNTRIES:
        return RuleTrigger(
            rule_name   = "foreign_transaction",
            description = f"Transaction in {txn.country} while home is {txn.home_country}"
                          + (" (high-risk)" if is_high_risk else ""),
            severity    = "high" if is_high_risk else "low",
            value       = 1.0,
            threshold   = 0.0,
        )
    return None


def rule_odd_hour(txn: TransactionRequest) -> Optional[RuleTrigger]:
    if 1 <= txn.hour <= 4 and txn.amount > txn.avg_user_spend * 3:
        return RuleTrigger(
            rule_name   = "odd_hour_large_transaction",
            description = f"Large transaction (EUR{txn.amount:.2f}) at {txn.hour:02d}:xx",
            severity    = "medium",
            value       = float(txn.hour),
            threshold   = 1.0,
        )
    return None


def rule_card_not_present_high_risk(txn: TransactionRequest) -> Optional[RuleTrigger]:
    if txn.is_card_present == 0 and txn.mcc in HIGH_RISK_MCC and txn.amount > 200:
        return RuleTrigger(
            rule_name   = "cnp_high_risk_mcc",
            description = f"Card-not-present at high-risk MCC {txn.mcc} for EUR{txn.amount:.2f}",
            severity    = "medium",
            value       = txn.amount,
            threshold   = 200.0,
        )
    return None


def rule_atm_foreign(txn: TransactionRequest) -> Optional[RuleTrigger]:
    if txn.mcc == 6011 and txn.country != txn.home_country:
        return RuleTrigger(
            rule_name   = "atm_foreign_country",
            description = f"ATM withdrawal in foreign country {txn.country}",
            severity    = "high",
            value       = 1.0,
            threshold   = 0.0,
        )
    return None


def rule_round_amount_cnp(txn: TransactionRequest) -> Optional[RuleTrigger]:
    if txn.amount % 100 == 0 and txn.amount >= 500 and txn.is_card_present == 0:
        return RuleTrigger(
            rule_name   = "round_amount_cnp",
            description = f"Round amount EUR{txn.amount:.0f} via card-not-present",
            severity    = "low",
            value       = txn.amount,
            threshold   = 500.0,
        )
    return None


ALL_RULES = [
    rule_high_value, rule_near_credit_limit, rule_foreign_transaction,
    rule_odd_hour, rule_card_not_present_high_risk, rule_atm_foreign, rule_round_amount_cnp,
]


def evaluate_rules(txn: TransactionRequest) -> list[RuleTrigger]:
    return [r for fn in ALL_RULES if (r := fn(txn)) is not None]


def rule_severity_score(rules: list[RuleTrigger]) -> float:
    if not rules:
        return 0.0
    weights = {"low": 0.15, "medium": 0.35, "high": 0.65}
    return min(1.0, sum(weights.get(r.severity, 0.2) for r in rules))
