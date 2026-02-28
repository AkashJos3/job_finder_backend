import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('c:/app/backend/.env')
url = os.environ.get('SUPABASE_URL')
service_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_KEY')
admin_supabase = create_client(url, service_key)

try:
    users = admin_supabase.auth.admin.list_users()
    if users:
        print("First user:", users[0].id)
        # Attempt to get user
        user = admin_supabase.auth.admin.get_user_by_id(users[0].id)
        print("User:", user)
        if hasattr(user.user, 'banned_until'):
            print("banned_until exists!")
            print(user.user.banned_until)
except Exception as e:
    import traceback
    traceback.print_exc()
