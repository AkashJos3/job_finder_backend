import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
client = create_client(url, key)

resp = client.table('jobs').select('*').limit(1).execute()
if resp.data:
    print("Keys:", resp.data[0].keys())
else:
    print("No data, but table exists.")
