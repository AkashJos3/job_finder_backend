import os
import requests
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

try:
    res = requests.get(f"{url}/rest/v1/", headers={'apikey': key})
    defs = res.json().get('definitions', {})
    shifts_def = defs.get('shifts', {})
    if shifts_def:
        print("Shifts table schema:")
        for col, details in shifts_def.get('properties', {}).items():
            print(f"  {col}: {details.get('type')}")
    else:
        print("No shifts definition found in openapi.")
except Exception as e:
    print("Error:", e)
