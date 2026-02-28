import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)

def seed_data():
    try:
        # 1. Get an employer and a student to use for references
        employers_res = supabase.table('profiles').select('id, full_name').eq('role', 'employer').limit(3).execute()
        students_res = supabase.table('profiles').select('id, full_name').eq('role', 'student').limit(3).execute()
        
        employers = employers_res.data
        students = students_res.data
        
        if not employers or not students:
            print("Need at least 1 employer and 1 student in the DB to seed admin data.")
            return

        print("Seeding employer verifications...")
        for emp in employers:
            # Check if verification already exists
            existing = supabase.table('employer_verifications').select('id').eq('employer_id', emp['id']).execute()
            if not existing.data:
                v_data = {
                    "employer_id": emp['id'],
                    "business_name": f"{emp.get('full_name', 'Tech')} Corp",
                    "registration_number": f"REG-{emp['id'][:8].upper()}",
                    "document_url": "https://example.com/doc.pdf",
                    "status": "pending",
                    "submitted_at": datetime.utcnow().isoformat()
                }
                supabase.table('employer_verifications').insert(v_data).execute()
                print(f"Added verification for {v_data['business_name']}")

        print("Seeding job reports...")
        jobs_res = supabase.table('jobs').select('id, title, employer_id').limit(2).execute()
        jobs = jobs_res.data
        
        if jobs and students:
            for i, job in enumerate(jobs):
                reporter = students[i % len(students)]
                # Check if report already exists
                existing = supabase.table('reports').select('id').eq('job_id', job['id']).execute()
                if not existing.data:
                    r_data = {
                        "job_id": job['id'],
                        "reporter_id": reporter['id'],
                        "reported_id": job['employer_id'],
                        "reason": ["Suspicious payment terms", "Inappropriate job description", "Spam / Fake Job"][i % 3],
                        "status": "pending"
                    }
                    supabase.table('reports').insert(r_data).execute()
                    print(f"Added report for job {job['id']}")
                    
        print("Data seeding completed successfully!")
    except Exception as e:
        print("Error seeding data:", e)

if __name__ == '__main__':
    seed_data()
