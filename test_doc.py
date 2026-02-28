import os
import requests
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

res = requests.get(f"{url}/rest/v1/merchant_verifications?select=document_url&limit=1", headers={'apikey': key, 'Authorization': f'Bearer {key}'})
data = res.json()
if data and 'document_url' in data[0]:
    val = data[0]['document_url']
    if val:
        print(f"Length: {len(val)}")
        print(f"Prefix: {val[:100]}...")
    else:
        print("Empty")
else:
    print("No data")
