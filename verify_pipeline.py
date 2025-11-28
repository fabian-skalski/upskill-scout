import requests
import time
import sys
import os
from redis import Redis
from rq.job import Job
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()

BACKEND_URL = "http://localhost:8000"
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")

def submit_job():
    print("Submitting job...")
    payload = {
        "description": "We are looking for a Python Engineering Manager with experience in FastAPI and Milvus vector and Snowflake database. " * 10, # Make it long enough to potentially be truncated in Redis/Milvus
        "sourceUrl": "https://example.com/job/12345",
        "timestamp": "2025-11-28T11:00:00Z",
    }
    try:
        response = requests.post(f"{BACKEND_URL}/text", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Job submitted successfully. Hash: {data['text_hash']}")
        return data['text_hash'], payload['description'], payload['sourceUrl'], payload['timestamp']
    except Exception as e:
        print(f"Failed to submit job: {e}")
        sys.exit(1)

def verify_job(text_hash, original_text, source_url, timestamp):
    print("Waiting for processing...")
    max_retries = 30
    retry_interval = 2 # seconds
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{BACKEND_URL}/job/{text_hash}")
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "completed":
                    print("Job processing completed! Verifying Milvus data...")
                    
                    # Verify Milvus data
                    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
                    
                    res = client.query(
                        collection_name=MILVUS_COLLECTION_NAME,
                        filter=f"text_hash == '{text_hash}'",
                        output_fields=["original_full_text", "source_url", "timestamp", "vector", "llm_inferred_title", "llm_inferred_skills"]
                    )
                    
                    if not res:
                        print("Verification FAILED! Job not found in Milvus.")
                        sys.exit(1)

                    job_data = res[0]
                    
                    # Check text integrity
                    retrieved_text = job_data.get("original_full_text")
                    if retrieved_text == original_text:
                        print(f"Text integrity check PASSED. Length: {len(retrieved_text)}")
                    else:
                        print(f"Text integrity check FAILED!")
                        print(f"Original length: {len(original_text)}")
                        print(f"Retrieved length: {len(retrieved_text) if retrieved_text else 0}")

                    # Check vector
                    vector = job_data.get("vector")
                    if vector and len(vector) > 0 and any(v != 0 for v in vector):
                        print("Embedding check PASSED (non-zero vector found).")
                    else:
                        print("Embedding check FAILED (vector missing or zero).")
                        sys.exit(1)

                    # Check new fields
                    title = job_data.get("llm_inferred_title")
                    skills = job_data.get("llm_inferred_skills")
                    retrieved_source_url = job_data.get("source_url")
                    retrieved_timestamp = job_data.get("timestamp")
                    
                    if title:
                        print(f"Title check PASSED: {title}")
                    else:
                        print("Title check WARNING (missing)")
                        
                    if skills and len(skills) > 0:
                        print(f"Skills check PASSED: Found {skills}")
                    else:
                        print("Skills check WARNING (missing or empty)")
                    
                    # Check source_url
                    if retrieved_source_url == source_url:
                        print(f"Source URL check PASSED: {retrieved_source_url}")
                    else:
                        print(f"Source URL check FAILED! Expected: {source_url}, Got: {retrieved_source_url}")
                        sys.exit(1)
                    
                    # Check timestamp
                    if retrieved_timestamp == timestamp:
                        print(f"Timestamp check PASSED: {retrieved_timestamp}")
                    else:
                        print(f"Timestamp check FAILED! Expected: {timestamp}, Got: {retrieved_timestamp}")
                        sys.exit(1)

                    print("Verification SUCCESS!")
                    return
                else:
                    print(f"Status: {data.get('status')} - {data.get('detail')}. Retrying ({i+1}/{max_retries})...")
            else:
                print(f"Backend Error: {response.status_code}. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        
        time.sleep(retry_interval)
    
    print("Verification FAILED: Timed out.")
    sys.exit(1)

if __name__ == "__main__":    
    hash_val, text, url, ts = submit_job()
    verify_job(hash_val, text, url, ts)
