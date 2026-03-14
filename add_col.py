import os
import requests

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

# Add submission_attempts column
query = """
ALTER TABLE employer_verifications 
ADD COLUMN IF NOT EXISTS submission_attempts INTEGER DEFAULT 1;
"""
res = requests.post(f"{url}/rest/v1/", headers=headers, json={"query": query})
print("Added column", res.status_code, res.text)
