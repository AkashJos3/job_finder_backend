import os
import requests
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
res = requests.get(f"{url}/rest/v1/jobs?limit=1", headers={'apikey': key, 'Authorization': f'Bearer {key}'})
data = res.json()
if isinstance(data, list) and len(data) > 0:
    print(data[0].keys())
else:
    print("Error or empty:", data)
