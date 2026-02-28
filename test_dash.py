import os
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('c:/app/backend/.env')
sb_url = os.environ.get('SUPABASE_URL')
sb_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_KEY')

supabase = create_client(sb_url, sb_key)

recent_res = supabase.table('profiles').select('id, full_name, role, created_at, is_verified').order('created_at', desc=True).limit(5).execute()
recent_profiles = recent_res.data
employer_ids = [p['id'] for p in recent_profiles if p['role'] == 'employer']

print("Employer IDs:", employer_ids)
if employer_ids:
    verifs = supabase.table('employer_verifications').select('employer_id, status').in_('employer_id', employer_ids).execute()
    verif_map = {v['employer_id']: v['status'] for v in verifs.data}
    print("Verification Map:", verif_map)
    for p in recent_profiles:
        if p['role'] == 'employer':
            p['verification_status'] = verif_map.get(p['id'], 'approved' if p.get('is_verified') else 'pending')

for p in recent_profiles:
    if p['role'] == 'employer':
        print(f"{p['full_name']} -> {p.get('verification_status')}")
