import os
import requests
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

res = requests.get(f"{url}/rest/v1/employer_verifications?select=id,business_name,document_url&order=submitted_at.desc", headers={'apikey': key, 'Authorization': f'Bearer {key}'})
data = res.json()
print(f"Total rows: {len(data)}")
for row in data:
    bname = row.get('business_name', '')
    url_val = row.get('document_url', '') or ''
    print(f"[{bname}] Length: {len(url_val)}")
    if url_val.startswith('data:'):
        print("  Format: Base64 Data URL")
    elif url_val.startswith('http'):
        print(f"  Format: HTTP Link -> {url_val[:50]}...")
    else:
        print("  Format: Unknown")
