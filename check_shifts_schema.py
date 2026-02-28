import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
client = create_client(url, key)

resp = client.table('shifts').select('*').limit(1).execute()
if resp.data:
    print("Keys in shifts table:", resp.data[0].keys())
else:
    print("Shifts table exists, but is empty. Trying an insert and quick delete or just checking schema...")
    # Insert a dummy record and delete it.
    try:
        # Just passing some required fields might fail if required, but we can see the error.
        insert_resp = client.table('shifts').insert({
            'employer_id': '617fd57b-b699-4cc1-9c7a-88f00c6b865f',
            'job_id': '06aca07c-cf10-4cfc-8030-aec0cc0375c4',
            'shift_date': '2026-03-01',
            'start_time': '09:00',
            'end_time': '17:00'
        }).execute()
        print("Insert succeeded with end_time! Keys:", insert_resp.data[0].keys())
        client.table('shifts').delete().eq('id', insert_resp.data[0]['id']).execute()
    except Exception as e:
        import traceback
        traceback.print_exc()

