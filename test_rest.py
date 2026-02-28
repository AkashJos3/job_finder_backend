import os
import requests
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
service_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or key

headers = {
    'apikey': service_key,
    'Authorization': f'Bearer {service_key}',
    'Content-Type': 'application/json'
}

# List users
res = requests.get(f"{url}/auth/v1/admin/users", headers=headers)
users = res.json().get('users', [])
if users:
    uid = users[0]['id']
    print(f"Banning {uid}...")
    # Update ban duration
    update_res = requests.put(f"{url}/auth/v1/admin/users/{uid}", headers=headers, json={"ban_duration": "8760h"})
    print("Ban response:", update_res.status_code, update_res.json())
    
    # Check again
    res2 = requests.get(f"{url}/auth/v1/admin/users", headers=headers)
    print("Banned until:", next(u for u in res2.json().get('users', []) if u['id'] == uid).get('banned_until'))
    
    # Unban
    update_res2 = requests.put(f"{url}/auth/v1/admin/users/{uid}", headers=headers, json={"ban_duration": "none"})
    print("Unban response:", update_res2.status_code, update_res2.json())
else:
    print(res.status_code, res.text)
