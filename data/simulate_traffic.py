import asyncio
import csv
import json
import random
import time
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).parent.parent
TEST_CSV = ROOT_DIR / "data" / "test.csv"
API_URL = "http://localhost:8000/predict"

async def send_request(client, payload):
    try:
        response = await client.post(API_URL, json=payload)
        return response.status_code
    except Exception as e:
        return str(e)

async def simulate_traffic(num_requests=2500, concurrency=50):
    # Load transactions
    transactions = []
    with open(TEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append({
                "transaction_id": row["transaction_id"],
                "user_id": int(row["user_id"]),
                "amount": float(row["amount"]),
                "mcc": int(row["mcc"]),
                "country": row["country"],
                "home_country": row["home_country"],
                "hour": int(row["hour"]),
                "day_of_week": int(row["day_of_week"]),
                "is_card_present": int(row["is_card_present"]),
                "credit_limit": float(row["credit_limit"]),
                "avg_user_spend": float(row["avg_user_spend"])
            })

    print(f"Loaded {len(transactions)} transactions from test.csv")
    
    # Select a random subset to reach num_requests
    sampled = random.choices(transactions, k=num_requests)
    
    print(f"Simulating {num_requests} requests with concurrency {concurrency}...")
    t0 = time.time()
    
    async with httpx.AsyncClient() as client:
        tasks = set()
        completed = 0
        
        for payload in sampled:
            if len(tasks) >= concurrency:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                completed += len(done)
                if completed % 500 == 0:
                    print(f"Sent {completed} requests...")
                    
            tasks.add(asyncio.create_task(send_request(client, payload)))
            
        if tasks:
            done, _ = await asyncio.wait(tasks)
            completed += len(done)
            
    elapsed = time.time() - t0
    print(f"Finished sending {completed} requests in {elapsed:.2f} seconds!")
    print(f"Throughput: {completed/elapsed:.1f} req/s")

if __name__ == "__main__":
    asyncio.run(simulate_traffic(num_requests=100000, concurrency=100))
