import os
import requests
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

res = requests.get(f"{url}/rest/v1/profiles?select=full_name,email,phone&limit=1", headers={'apikey': key, 'Authorization': f'Bearer {key}'})
data = res.json()
print("PROFILES DATA:", data)
