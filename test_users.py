import requests
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = "http://localhost:5000/api/admin/users"
sb_url = os.environ.get('SUPABASE_URL')
sb_key = os.environ.get('SUPABASE_KEY')

supabase = create_client(sb_url, sb_key)
res = supabase.auth.sign_in_with_password({'email': 'admin@afterbell.com', 'password': 'password123'})
token = res.session.access_token

response = requests.get(url, headers={'Authorization': f'Bearer {token}'})
data = response.json()
print("USERS DATA:")
for u in data.get('data', []):
    if u['role'] == 'employer':
        print(f"{u['full_name']} -> Active Jobs: {u.get('activeJobs')}, Total Hires: {u.get('totalHires')}, Loc: {u.get('location')} Address: {u.get('address')}")
