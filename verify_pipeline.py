import requests
import time
import sys

from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000"

def submit_job():
    print("Submitting job...")
    payload = {
        "text": "We are looking for a Senior Python Developer with experience in FastAPI and Milvus.",
        "user_id": "test_user_1"
    }
    try:
        response = requests.post(f"{API_URL}/text", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Job submitted successfully. Hash: {data['text_hash']}")
        return data['text_hash']
    except Exception as e:
        print(f"Failed to submit job: {e}")
        sys.exit(1)

def verify_job(text_hash):
    print("Waiting for processing...")
    max_retries = 30
    retry_interval = 10 # seconds
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/job/{text_hash}")
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "completed":
                    print(f"Verification SUCCESS! Data: {data['data']}")
                    
                    # Check embeddings
                    embedding = data['data'].get('vector') or data['data'].get('text_embedding')
                    if not embedding:
                        print("FAILED: No embedding found in data.")
                        sys.exit(1)
                        
                    if all(v == 0.0 for v in embedding):
                        print("FAILED: Embedding is all zeros.")
                        sys.exit(1)
                        
                    print("Embedding check PASSED (non-zero vector found).")
                    return
                else:
                    print(f"Status: {data.get('status')} - {data.get('detail')}. Retrying ({i+1}/{max_retries})...")
            else:
                print(f"API Error: {response.status_code}. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        
        time.sleep(retry_interval)
    
    print("Verification FAILED: Timed out.")
    sys.exit(1)

if __name__ == "__main__":    
    hash_val = submit_job()
    verify_job(hash_val)
