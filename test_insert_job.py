import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
client = create_client(url, key)

try:
    resp = client.table('jobs').insert({
        "employer_id": "617fd57b-b699-4cc1-9c7a-88f00c6b865f",
        "title": "Test Title",
        "company_name": "Test Company",
        "wage": "500",
        "location": "NY",
        "latitude": 40.71,
        "longitude": -74.0,
        "description": "Test",
        "requirements": "Test",
        "urgent": False,
        "status": "open",
        "image_url": "test_url"
    }).execute()
    print("Success:", resp.data)
except Exception as e:
    import traceback
    traceback.print_exc()
