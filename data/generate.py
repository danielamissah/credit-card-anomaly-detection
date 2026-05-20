import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

RANDOM_SEED = 42
N_USERS     = 500
N_NORMAL    = 50_000
N_ANOMALIES = 2_500
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MCC = {"grocery":5411,"restaurant":5812,"gas":5541,"online":5999,
       "travel":4722,"entertainment":7922,"atm":6011,"pharmacy":5912,"electronics":5734,"luxury":5944}
MCC_LIST = list(MCC.values())


def gen_profiles(n):
    countries = ["DE","DE","DE","DE","AT","CH","FR","NL","GB","US"]
    return pd.DataFrame([{
        "user_id": i,
        "avg_spend": np.random.lognormal(3.5, 0.8),
        "home_country": random.choice(countries),
        "active_hours": random.choice(["morning","daytime","evening"]),
        "preferred_mccs": random.sample(MCC_LIST, k=random.randint(3,6)),
        "credit_limit": np.random.choice([1000,2000,5000,10000,20000]),
    } for i in range(n)])


def hour_for(profile):
    m = {"morning":9,"daytime":13,"evening":19}
    return int(np.clip(np.random.normal(m[profile], 2), 0, 23))


def make_row(idx, user, dt, amount, mcc, country, is_anomaly, atype):
    return {
        "transaction_id": f"TXN_{idx:07d}", "user_id": user["user_id"],
        "timestamp": dt, "amount": round(amount, 2), "mcc": mcc,
        "country": country, "home_country": user["home_country"],
        "hour": dt.hour, "day_of_week": dt.weekday(),
        "is_card_present": 1 if random.random() > 0.3 else 0,
        "credit_limit": user["credit_limit"], "avg_user_spend": user["avg_spend"],
        "is_anomaly": is_anomaly, "anomaly_type": atype,
    }


def gen_normal(profiles, n, start):
    rows = []
    for _ in range(n):
        u  = profiles.sample(1).iloc[0]
        dt = start + timedelta(days=random.randint(0,364), hours=hour_for(u["active_hours"]), minutes=random.randint(0,59))
        rows.append(make_row(len(rows), u, dt, max(0.5, np.random.lognormal(np.log(u["avg_spend"]), 0.5)),
                             random.choice(u["preferred_mccs"]), u["home_country"], 0, "normal"))
    return pd.DataFrame(rows)


def gen_anomalies(profiles, n, start):
    distant = ["CN","NG","RU","BR","VN","UA"]
    rows = []
    n4   = n // 4

    for i in range(n4):
        u  = profiles.sample(1).iloc[0]
        dt = start + timedelta(days=random.randint(0,364), hours=random.randint(0,23))
        rows.append(make_row(len(rows), u, dt, min(u["avg_spend"]*random.uniform(15,50), u["credit_limit"]*0.95),
                             MCC["luxury"], u["home_country"], 1, "high_value"))

    for i in range(n4 // 8):
        u    = profiles.sample(1).iloc[0]
        base = start + timedelta(days=random.randint(0,364), hours=random.randint(2,5))
        for j in range(8):
            rows.append(make_row(len(rows), u, base+timedelta(minutes=j*3),
                                 round(random.uniform(1,9.99),2), MCC["online"], u["home_country"], 1, "velocity"))

    for i in range(n4):
        u  = profiles.sample(1).iloc[0]
        dt = start + timedelta(days=random.randint(0,364), hours=random.randint(0,23))
        rows.append(make_row(len(rows), u, dt, max(0.5, np.random.lognormal(np.log(u["avg_spend"]),0.5)),
                             random.choice(MCC_LIST), random.choice(distant), 1, "geographic"))

    for i in range(n4):
        u  = profiles[profiles["active_hours"] != "morning"].sample(1).iloc[0]
        dt = start + timedelta(days=random.randint(0,364), hours=random.randint(3,5))
        rows.append(make_row(len(rows), u, dt, u["avg_spend"]*random.uniform(3,10),
                             MCC["atm"], u["home_country"], 1, "odd_hour"))

    return pd.DataFrame(rows)


def add_features(df):
    df = df.copy()
    df["timestamp"]           = pd.to_datetime(df["timestamp"])
    df["log_amount"]          = np.log1p(df["amount"])
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_user_spend"] + 1e-6)
    df["amount_to_limit_ratio"] = df["amount"] / (df["credit_limit"] + 1e-6)
    df["is_foreign"]          = (df["country"] != df["home_country"]).astype(int)
    df["hour_bin"]            = pd.cut(df["hour"], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
    df["is_weekend"]          = (df["day_of_week"] >= 5).astype(int)
    mcc_r = {5411:0.1,5812:0.1,5541:0.2,5999:0.4,4722:0.3,7922:0.2,6011:0.6,5912:0.1,5734:0.5,5944:0.7}
    df["mcc_risk_score"]      = df["mcc"].map(mcc_r).fillna(0.3)
    df["velocity_1h"]         = 1
    return df


def main():
    start    = datetime(2023, 1, 1)
    profiles = gen_profiles(N_USERS)
    print(f"Generating {N_NORMAL:,} normal transactions ...")
    normal   = gen_normal(profiles, N_NORMAL, start)
    print(f"Injecting {N_ANOMALIES:,} anomalies ...")
    anomalies = gen_anomalies(profiles, N_ANOMALIES, start)
    full     = add_features(pd.concat([normal, anomalies], ignore_index=True).sort_values("timestamp").reset_index(drop=True))
    normal_o = full[full["is_anomaly"] == 0]
    train    = normal_o.sample(frac=0.8, random_state=RANDOM_SEED)
    test     = pd.concat([
        normal_o.drop(train.index).sample(n=1000, random_state=RANDOM_SEED),
        full[full["is_anomaly"] == 1].sample(n=500, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED)
    full.to_csv(f"{OUTPUT_DIR}/transactions.csv",  index=False)
    train.to_csv(f"{OUTPUT_DIR}/train.csv",         index=False)
    test.to_csv(f"{OUTPUT_DIR}/test.csv",           index=False)
    print(f"\n Dataset generated:")
    print(f"   Full:  {len(full):,} rows | Anomaly rate: {full['is_anomaly'].mean():.1%}")
    print(f"   Train: {len(train):,} rows (normal only)")
    print(f"   Test:  {len(test):,} rows")

if __name__ == "__main__":
    main()
