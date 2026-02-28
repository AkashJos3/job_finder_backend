import os
import requests
import math
import io
import base64
import json
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, request
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import PIL.Image
from google import genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for the React frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase: Client = create_client(url, key)

# --- MIDDLEWARE ---
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized', 'message': 'Missing or invalid token'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            # Verify JWT using Supabase
            user_res = supabase.auth.get_user(token)
            if not user_res or not user_res.user:
                return jsonify({'error': 'Unauthorized', 'message': 'Invalid token'}), 401
            
            # Attach user to request
            request.user = user_res.user
        except Exception as e:
            return jsonify({'error': 'Unauthorized', 'message': str(e)}), 401
            
        return f(*args, **kwargs)
    return decorated

# --- GEOCODING & DISTANCE HELPERS ---

def geocode_location(address):
    """Convert a text location into (latitude, longitude) using OpenStreetMap."""
    if not address or not isinstance(address, str):
        return None, None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "AfterBell/1.0 (student-jobs-portal)"}
        params = {"q": address, "format": "json", "limit": 1}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.ok:
            data = response.json()
            if data and len(data) > 0:
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print(f"Geocoding failed for {address}: {e}")
    return None, None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0 # Radius of earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- GENERAL ---
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple connection test endpoint"""
    return jsonify({
        "status": "online",
        "message": "Flask backend is running and connected to Supabase!"
    }), 200

# --- JOBS API ---

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Fetch jobs, optionally sorted by distance if lat/lng provided"""
    try:
        user_lat = request.args.get('lat', type=float)
        user_lng = request.args.get('lng', type=float)
        
        # Only show open jobs to students
        res = supabase.table('jobs').select('*, profiles!jobs_employer_id_fkey(full_name, avatar_url, company_name)').eq('status', 'open').execute()
        jobs = res.data or []
        
        # Sort by distance if coordinates provided
        if user_lat is not None and user_lng is not None:
            for job in jobs:
                j_lat = job.get('latitude')
                j_lng = job.get('longitude')
                dist = haversine_distance(user_lat, user_lng, j_lat, j_lng)
                if dist is not None:
                    job['distance_km'] = round(dist, 2)
            # Sort: jobs with distance first (closest), then jobs without coordinates
            jobs.sort(key=lambda x: (x.get('distance_km') is None, x.get('distance_km', 99999), x.get('created_at', '')), reverse=False)
        else:
            # Default sort by newest if no location
            jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
        return jsonify({"data": jobs}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/employer/jobs', methods=['GET'])
@require_auth
def get_employer_jobs():
    """Fetch all jobs posted by the logged-in employer"""
    try:
        employer_id = request.user.id
        # Get jobs and count applications for each job
        res = supabase.table('jobs').select('*, applications(count)').eq('employer_id', employer_id).order('created_at', desc=True).execute()
        return jsonify({"data": res.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/parse-image', methods=['POST'])
@require_auth
def parse_job_poster():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
            
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY not configured on server"}), 500
            
        client = genai.Client(api_key=api_key)

        # Remove header from base64 string if present (e.g. data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        prompt = """
        Extract the following information from this job poster image.
        Return ONLY a raw JSON object with these exact keys:
        - title: The job title (e.g. "Cashier", "Tutor")
        - description: A brief summary of the role and responsibilities.
        - wage: The pay or wage mentioned, just the number (e.g. "500"). If not found, use "".
        - requirements: Any shift timings or requirements mentioned (e.g. "Weekends only", "Morning shift").
        Do not include markdown formatting or json code blocks, just raw JSON.
        """
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt, img]
        )
        
        # Clean response in case it has markdown markers
        text = response.text.replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(text)
        
        return jsonify({"data": parsed_data}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
            
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs', methods=['POST'])
@require_auth
def create_job():
    """Create a new job post (Employer only, must be verified)"""
    try:
        data = request.json
        employer_id = request.user.id
        
        # Verify user is an employer AND is verified
        profile_res = supabase.table('profiles').select('role, is_verified').eq('id', employer_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'employer':
            return jsonify({"error": "Forbidden", "message": "Only employers can post jobs"}), 403
            
        if not profile_res.data[0].get('is_verified', False):
            return jsonify({"error": "Forbidden", "message": "Your employer account must be verified by an admin before posting jobs."}), 403

        location = data.get("location")
        lat, lng = geocode_location(location)

        job_data = {
            "employer_id": employer_id,
            "title": data.get("title"),
            "company_name": data.get("company_name"),
            "wage": data.get("wage"),
            "location": location,
            "latitude": lat,
            "longitude": lng,
            "description": data.get("description"),
            "requirements": data.get("requirements"),
            "urgent": data.get("urgent", False),
            "image_url": data.get("image_url"),
            "status": "open"
        }
        
        response = supabase.table('jobs').insert(job_data).execute()
        return jsonify({"data": response.data[0]}), 201
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"message": str(e), "error": str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['PUT'])
@require_auth
def update_job(job_id):
    """Update a specific job (must be owner)"""
    try:
        user_id = request.user.id
        data = request.json
        
        # Verify ownership
        job_res = supabase.table('jobs').select('employer_id').eq('id', job_id).execute()
        if not job_res.data or job_res.data[0]['employer_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403
            
        allowed_fields = ['title', 'description', 'location', 'wage', 'requirements', 'urgent', 'status', 'image_url']
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        # Re-geocode if location changed
        if 'location' in update_data:
            lat, lng = geocode_location(update_data['location'])
            update_data['latitude'] = lat
            update_data['longitude'] = lng
            
        res = supabase.table('jobs').update(update_data).eq('id', job_id).execute()
        return jsonify({"message": "Job updated successfully", "data": res.data[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
@require_auth
def delete_job(job_id):
    """Delete a specific job (must be owner)"""
    try:
        user_id = request.user.id
        
        # Verify ownership
        job_res = supabase.table('jobs').select('employer_id').eq('id', job_id).execute()
        if not job_res.data or job_res.data[0]['employer_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403
            
        # Supabase will cascade delete applications if foreign key supports it, 
        # otherwise we manually delete apps first
        supabase.table('applications').delete().eq('job_id', job_id).execute()
        supabase.table('saved_jobs').delete().eq('job_id', job_id).execute()
        res = supabase.table('jobs').delete().eq('id', job_id).execute()
        
        return jsonify({"message": "Job deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- SAVED JOBS API ---

@app.route('/api/jobs/save', methods=['POST'])
@require_auth
def save_job():
    """Toggle saving a job for a student"""
    try:
        data = request.json
        job_id = data.get('job_id')
        user_id = request.user.id
        
        if not job_id:
            return jsonify({"error": "Job ID required"}), 400
            
        # Check if already saved
        existing = supabase.table('saved_jobs').select('*').eq('student_id', user_id).eq('job_id', job_id).execute()
        
        if existing.data:
            # Unsave
            supabase.table('saved_jobs').delete().eq('student_id', user_id).eq('job_id', job_id).execute()
            return jsonify({"message": "Job removed from saved list", "saved": False}), 200
        else:
            # Save
            res = supabase.table('saved_jobs').insert({'student_id': user_id, 'job_id': job_id}).execute()
            return jsonify({"message": "Job saved successfully", "saved": True, "data": res.data[0]}), 201
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs/saved', methods=['GET'])
@require_auth
def get_saved_jobs():
    """Get all saved jobs for a student"""
    try:
        user_id = request.user.id
        # We need to fetch the saved_jobs and join with the jobs table
        # Explicit foreign key required due to multiple links to profiles table
        res = supabase.table('saved_jobs').select('*, jobs(*, profiles!jobs_employer_id_fkey(full_name, avatar_url))').eq('student_id', user_id).execute()
        return jsonify({"data": res.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- PROFILE API ---

@app.route('/api/profile', methods=['GET'])
@require_auth
def get_profile():
    """Get the current user's profile data"""
    try:
        user_id = request.user.id
        res = supabase.table('profiles').select('*').eq('id', user_id).execute()
        if not res.data:
            return jsonify({"error": "Profile not found"}), 404
        return jsonify({"data": res.data[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile', methods=['PUT'])
@require_auth
def update_profile():
    """Update the current user's profile data"""
    try:
        user_id = request.user.id
        data = request.json
        print(f"[UPDATE_PROFILE] user={user_id} data={data}")
        
        # Allow updating all fields we track in the frontend
        allowed_fields = ['full_name', 'avatar_url', 'phone', 'university', 'course', 'year', 'address', 'bio', 'company_name']
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        # Geocode the address if it was updated
        if 'address' in update_data and update_data['address']:
            lat, lng = geocode_location(update_data['address'])
            if lat is not None and lng is not None:
                update_data['latitude'] = lat
                update_data['longitude'] = lng

        if not update_data:
            return jsonify({"message": "No valid fields to update", "data": {}}), 200
            
        res = supabase.table('profiles').update(update_data).eq('id', user_id).execute()
        print(f"[UPDATE_PROFILE] Result: {res.data}")
        
        updated_row = res.data[0] if res.data else {}
        return jsonify({"message": "Profile updated successfully", "data": updated_row}), 200
    except Exception as e:
        import traceback
        print(f"[UPDATE_PROFILE] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- MESSAGES API ---

@app.route('/api/messages', methods=['POST'])
@require_auth
def send_message():
    """Send a message to another user"""
    try:
        data = request.json
        sender_id = request.user.id
        receiver_id = data.get('receiver_id')
        content = data.get('content')
        
        if not receiver_id or not content:
            return jsonify({"error": "receiver_id and content are required"}), 400
        
        print(f"[SEND_MSG] sender={sender_id} receiver={receiver_id} content={content[:30]}")
        res = supabase.table('messages').insert({
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'content': content
        }).execute()
        print(f"[SEND_MSG] Result: {res.data}")
        if res.data:
            return jsonify({"message": "Message sent", "data": res.data[0]}), 201
        else:
            return jsonify({"error": "Insert returned no data – check if messages table exists and RLS allows inserts"}), 500
    except Exception as e:
        import traceback
        print(f"[SEND_MSG] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/messages/<other_user_id>', methods=['GET'])
@require_auth
def get_messages(other_user_id):
    """Get chat history with another user"""
    try:
        user_id = request.user.id
        try:
            res = supabase.table('messages').select('*').or_(f"and(sender_id.eq.{user_id},receiver_id.eq.{other_user_id}),and(sender_id.eq.{other_user_id},receiver_id.eq.{user_id})").order('created_at').execute()
            return jsonify({"data": res.data}), 200
        except Exception:
            res1 = supabase.table('messages').select('*').eq('sender_id', user_id).eq('receiver_id', other_user_id).execute()
            res2 = supabase.table('messages').select('*').eq('sender_id', other_user_id).eq('receiver_id', user_id).execute()
            all_msgs = res1.data + res2.data
            all_msgs.sort(key=lambda x: x['created_at'])
            return jsonify({"data": all_msgs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/messages/conversations', methods=['GET'])
@require_auth
def get_conversations():
    """Get a list of users the current user has chatted with"""
    try:
        user_id = request.user.id
        res1 = supabase.table('messages').select('receiver_id, profiles!messages_receiver_id_fkey(full_name, avatar_url)').eq('sender_id', user_id).execute()
        res2 = supabase.table('messages').select('sender_id, profiles!messages_sender_id_fkey(full_name, avatar_url)').eq('receiver_id', user_id).execute()
        
        convos = {}
        for m in res1.data:
            r_id = m['receiver_id']
            if r_id not in convos:
                convos[r_id] = m.get('profiles') or {}
                convos[r_id]['id'] = r_id
        for m in res2.data:
            s_id = m['sender_id']
            if s_id not in convos:
                convos[s_id] = m.get('profiles') or {}
                convos[s_id]['id'] = s_id
        return jsonify({"data": list(convos.values())}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- SHIFTS API ---

@app.route('/api/shifts', methods=['POST'])
@require_auth
def create_shift():
    """Employer creates a new shift and assigns a student"""
    try:
        employer_id = request.user.id
        data = request.json
        job_id = data.get('job_id')
        student_id = data.get('student_id')  # optional
        shift_date = data.get('shift_date')  # YYYY-MM-DD
        start_time = data.get('start_time')  # HH:MM
        end_time = data.get('end_time')      # HH:MM
        notes = data.get('notes', '')

        if not job_id or not shift_date or not start_time or not end_time:
            return jsonify({'error': 'job_id, shift_date, start_time, end_time are required'}), 400

        # Verify ownership of job
        job_res = supabase.table('jobs').select('employer_id').eq('id', job_id).execute()
        if not job_res.data or job_res.data[0]['employer_id'] != employer_id:
            return jsonify({'error': 'Unauthorized'}), 403

        shift_data = {
            'employer_id': employer_id,
            'job_id': job_id,
            'student_id': student_id,
            'shift_date': shift_date,
            'start_time': start_time,
            'end_time': end_time,
            'status': 'Pending',
            'notes': notes,
        }
        res = supabase.table('shifts').insert(shift_data).execute()
        if student_id:
            create_notification(student_id, f"You have a new shift on {shift_date} ({start_time}–{end_time})", "shift")
        return jsonify({'message': 'Shift created successfully', 'data': res.data[0]}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shifts/employer', methods=['GET'])
@require_auth
def get_employer_shifts():
    """Fetch all shifts created by the employer"""
    try:
        employer_id = request.user.id
        res = supabase.table('shifts').select('*, profiles!shifts_student_id_fkey(full_name, avatar_url), jobs(title, location)').eq('employer_id', employer_id).order('shift_date', desc=False).execute()
        return jsonify({'data': res.data or []}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shifts/student', methods=['GET'])
@require_auth
def get_student_shifts():
    """Fetch all shifts assigned to the student"""
    try:
        student_id = request.user.id
        res = supabase.table('shifts').select('*, profiles!shifts_employer_id_fkey(full_name, avatar_url, company_name), jobs(title, location)').eq('student_id', student_id).order('shift_date', desc=False).execute()
        return jsonify({'data': res.data or []}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shifts/accepted-applicants', methods=['GET'])
@require_auth
def get_accepted_applicants():
    """Get accepted applicants for all employer's jobs (to assign to shifts)."""
    try:
        employer_id = request.user.id
        jobs_res = supabase.table('jobs').select('id, title').eq('employer_id', employer_id).execute()
        job_ids = [j['id'] for j in (jobs_res.data or [])]
        if not job_ids:
            return jsonify({'data': []}), 200
        apps_res = (
            supabase.table('applications')
            .select('id, job_id, student_id, status, jobs(title), profiles(full_name)')
            .in_('job_id', job_ids)
            .eq('status', 'accepted')
            .execute()
        )
        return jsonify({'data': apps_res.data or []}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shifts/<shift_id>', methods=['PUT'])
@require_auth
def update_shift_status(shift_id):
    """Update a shift's status or fields"""
    try:
        user_id = request.user.id
        data = request.json
        allowed = {k: v for k, v in data.items() if k in ('status', 'notes', 'student_id', 'start_time', 'end_time', 'shift_date')}
        if not allowed:
            return jsonify({'error': 'No valid fields provided'}), 400
        res = supabase.table('shifts').update(allowed).eq('id', shift_id).execute()
        if res.data and res.data[0].get('student_id') and 'status' in allowed:
            create_notification(res.data[0]['student_id'], f"Your shift status was updated to {allowed['status']}", "shift")
        return jsonify({'message': 'Shift updated', 'data': res.data[0] if res.data else {}}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shifts/<shift_id>', methods=['DELETE'])
@require_auth
def delete_shift(shift_id):
    """Delete a shift"""
    try:
        supabase.table('shifts').delete().eq('id', shift_id).execute()
        return jsonify({'message': 'Shift deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- REVIEWS API ---

@app.route('/api/reviews', methods=['POST'])
@require_auth
def create_review():
    """Employer rates a student after a shift"""
    try:
        employer_id = request.user.id
        data = request.json
        student_id = data.get('student_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        if not student_id or not rating:
            return jsonify({"error": "Missing student_id or rating"}), 400
            
        # Optional: verify they actually had a completed shift together
        # For MVP, we just insert the review
        review_data = {
            "employer_id": employer_id,
            "student_id": student_id,
            "rating": rating,
            "comment": comment
        }
        res = supabase.table('reviews').insert(review_data).execute()
        create_notification(student_id, f"You received a new {rating}-star rating from an employer", "review")
        return jsonify({"message": "Review submitted completely", "data": res.data[0]}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reviews/student/<student_id>', methods=['GET'])
@require_auth
def get_student_reviews(student_id):
    """Fetch all reviews for a student and calculate the average"""
    try:
        # fetch reviews with employer profile
        res = supabase.table('reviews').select('*, profiles!reviews_employer_id_fkey(full_name, company_name)').eq('student_id', student_id).order('created_at', desc=True).execute()
        
        reviews = res.data
        average_rating = 0
        if reviews:
            total = sum([r['rating'] for r in reviews])
            average_rating = round(total / len(reviews), 1)
            
        return jsonify({"data": reviews, "average": average_rating, "count": len(reviews)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NOTIFICATIONS API ---

def create_notification(user_id, message, type='general'):
    """Helper to create a notification"""
    try:
        supabase.table('notifications').insert({
            "user_id": user_id,
            "message": message,
            "type": type,
            "is_read": False
        }).execute()
    except Exception as e:
        print(f"Failed to create notification: {e}")

@app.route('/api/notifications', methods=['GET'])
@require_auth
def get_notifications():
    """Fetch notifications for logged in user"""
    try:
        user_id = request.user.id
        res = supabase.table('notifications').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(20).execute()
        return jsonify({"data": res.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications/<notif_id>/read', methods=['PUT'])
@require_auth
def mark_notification_read(notif_id):
    """Mark a notification as read"""
    try:
        user_id = request.user.id
        res = supabase.table('notifications').update({"is_read": True}).eq('id', notif_id).eq('user_id', user_id).execute()
        return jsonify({"message": "Marked as read", "data": res.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- APPLICATIONS API ---

@app.route('/api/applications', methods=['POST'])
@require_auth
def apply_job():
    """Student applies for a job"""
    try:
        data = request.json
        student_id = request.user.id
        job_id = data.get('job_id')
        cover_letter = data.get('cover_letter', None)
        
        if not job_id:
            return jsonify({"error": "Missing job_id"}), 400
            
        # Verify user is a student
        profile_res = supabase.table('profiles').select('role').eq('id', student_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'student':
            return jsonify({"error": "Forbidden", "message": "Only students can apply"}), 403
            
        app_data = {
            "job_id": job_id,
            "student_id": student_id,
            "status": "pending"
        }
        
        if cover_letter:
            app_data['cover_letter'] = cover_letter
        
        response = supabase.table('applications').insert(app_data).execute()
        
        job_res = supabase.table('jobs').select('employer_id, title').eq('id', job_id).execute()
        if job_res.data:
            employer_id = job_res.data[0]['employer_id']
            job_title = job_res.data[0]['title']
            
            student_res = supabase.table('profiles').select('full_name').eq('id', student_id).execute()
            student_name = student_res.data[0]['full_name'] if student_res.data else "A student"
            
            create_notification(employer_id, f"{student_name} applied for {job_title}", "application")

        return jsonify({"data": response.data[0]}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ADVANCED APPLICATIONS API ---

@app.route('/api/applications/student', methods=['GET'])
@require_auth
def get_student_applications():
    """Student fetches all their applications with job details"""
    try:
        student_id = request.user.id
        
        # Verify user is a student
        profile_res = supabase.table('profiles').select('role').eq('id', student_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'student':
            return jsonify({"error": "Forbidden", "message": "Only students can view their applications"}), 403
            
        # Fetch applications with associated job details
        # Must specify explicit foreign key for jobs->profiles since there are multiple links between the two tables
        response = supabase.table('applications').select('*, jobs(*, profiles!jobs_employer_id_fkey(full_name))').eq('student_id', student_id).order('created_at', desc=True).execute()
        return jsonify({"data": response.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/check-email', methods=['POST'])
def check_email_exists():
    """Validates if an email is registered in the system to prevent OTP bypass signup"""
    try:
        data = request.json
        email = data.get('email', '').lower()
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        service_role_key = os.getenv('SUPABASE_KEY')
        supabase_url = os.getenv('SUPABASE_URL')
        
        headers = {
            'apikey': service_role_key,
            'Authorization': f'Bearer {service_role_key}'
        }
        
        # We can fetch all users or use the admin rest endpoint.
        # But for scaling, better to check the profiles table if they have a role populated.
        # Wait, the easiest way to check if an account is strictly valid is grabbing the profiles table.
        # However, profiles table doesn't have email. So we fetch users from auth backend.
        users_res = requests.get(f'{supabase_url}/auth/v1/admin/users', headers=headers)
        if users_res.status_code == 200:
            for u in users_res.json().get('users', []):
                if u.get('email', '').lower() == email:
                    return jsonify({"exists": True}), 200
                    
        return jsonify({"exists": False}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/applications/employer', methods=['GET'])
@require_auth
def get_employer_applications():
    """Employer fetches all applicants for their jobs"""
    try:
        employer_id = request.user.id
        print(f"[EMPLOYER_APPS] employer_id={employer_id}")
        
        # Verify user is an employer
        profile_res = supabase.table('profiles').select('role').eq('id', employer_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'employer':
            return jsonify({"error": "Forbidden", "message": "Only employers can view applicants"}), 403
            
        # Step 1: Get all job IDs that belong to this employer
        print("[EMPLOYER_APPS] Step 1: fetching jobs")
        jobs_res = supabase.table('jobs').select('id, title, location, wage, company_name').eq('employer_id', employer_id).execute()
        print(f"[EMPLOYER_APPS] Jobs found: {len(jobs_res.data) if jobs_res.data else 0}")
        if not jobs_res.data:
            return jsonify({"data": []}), 200
        
        job_ids = [j['id'] for j in jobs_res.data]
        jobs_map = {j['id']: j for j in jobs_res.data}
        
        # Step 2: Fetch applications for those jobs
        print(f"[EMPLOYER_APPS] Step 2: fetching apps for job_ids={job_ids}")
        response = supabase.table('applications').select('*').in_('job_id', job_ids).order('created_at', desc=True).execute()
        apps = response.data or []
        print(f"[EMPLOYER_APPS] Applications found: {len(apps)}")
        
        if not apps:
            return jsonify({"data": []}), 200
        
        # Step 3: Fetch student profiles separately
        student_ids = list(set(a['student_id'] for a in apps))
        print(f"[EMPLOYER_APPS] Step 3: fetching profiles for {len(student_ids)} students")
        profiles_res = supabase.table('profiles').select('id, full_name, avatar_url, phone, university, course, year, bio, address').in_('id', student_ids).execute()
        profiles_map = {p['id']: p for p in (profiles_res.data or [])}
        
        # Step 4: Enrich each application with job and student profile data
        print("[EMPLOYER_APPS] Step 4: enriching data")
        enriched = []
        for app in apps:
            app['jobs'] = jobs_map.get(app['job_id'], {})
            app['profiles'] = profiles_map.get(app['student_id'], {})
            enriched.append(app)
        
        print(f"[EMPLOYER_APPS] Returning {len(enriched)} enriched applications")
        return jsonify({"data": enriched}), 200
    except Exception as e:
        import traceback
        print(f"[EMPLOYER_APPS] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/applications/status', methods=['POST'])
@require_auth
def update_application_status():
    """Employer updates the status of an application (accept/reject)"""
    try:
        data = request.json
        employer_id = request.user.id
        application_id = data.get('application_id')
        new_status = data.get('status')
        
        if not application_id or not new_status:
            return jsonify({"error": "Missing application_id or status"}), 400
            
        # First verify the application belongs to a job owned by this employer
        app_res = supabase.table('applications').select('*, jobs!inner(employer_id, title)').eq('id', application_id).execute()
        
        if not app_res.data or len(app_res.data) == 0:
             return jsonify({"error": "Not Found", "message": "Application not found"}), 404
             
        if app_res.data[0]['jobs']['employer_id'] != employer_id:
            return jsonify({"error": "Forbidden", "message": "You do not own this job"}), 403
            
        # Update the status
        update_res = supabase.table('applications').update({'status': new_status}).eq('id', application_id).execute()
        
        student_id = app_res.data[0]['student_id']
        job_title = app_res.data[0]['jobs'].get('title', 'a job')
        create_notification(student_id, f"Your application for {job_title} was {new_status}", "status_update")
        
        return jsonify({"data": update_res.data[0], "message": f"Application {new_status} successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ADMIN / REPORTS & USERS API ---

@app.route('/api/admin/dashboard/stats', methods=['GET'])
@require_auth
def get_dashboard_stats():
    """Admin fetches summary dashboard statistics"""
    try:
        admin_id = request.user.id
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden"}), 403

        # Pending Approvals (employers waiting for verification)
        pending_res = supabase.table('employer_verifications').select('id', count='exact').eq('status', 'pending').execute()
        pending_count = pending_res.count if pending_res.count is not None else len(pending_res.data)

        # Active Jobs
        jobs_res = supabase.table('jobs').select('id', count='exact').execute()
        jobs_count = jobs_res.count if jobs_res.count is not None else len(jobs_res.data)

        # Total Users (excluding admin)
        users_res = supabase.table('profiles').select('id', count='exact').neq('role', 'admin').execute()
        users_count = users_res.count if users_res.count is not None else len(users_res.data)

        stats = {
            "pending": pending_count,
            "jobs": jobs_count,
            "users": users_count
        }
        
        # Recent Activity (newest users)
        recent_res = supabase.table('profiles').select('id, full_name, role, created_at, is_verified').order('created_at', desc=True).limit(5).execute()
        recent_profiles = recent_res.data
        
        # Add verification status for employers
        employer_ids = [p['id'] for p in recent_profiles if p['role'] == 'employer']
        if employer_ids:
            verifs = supabase.table('employer_verifications').select('employer_id, status').in_('employer_id', employer_ids).execute()
            verif_map = {v['employer_id']: v['status'] for v in verifs.data}
            for p in recent_profiles:
                if p['role'] == 'employer':
                    p['verification_status'] = verif_map.get(p['id'], 'approved' if p.get('is_verified') else 'pending')

        return jsonify({"data": {"stats": stats, "recentActivity": recent_profiles}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports', methods=['POST'])
@require_auth
def submit_report():
    """Student/User reports a job or another user"""
    try:
        data = request.json
        reporter_id = request.user.id
        reported_id = data.get('reported_id')
        job_id = data.get('job_id')
        reason = data.get('reason')
        
        if not reason:
            return jsonify({"error": "Missing reason"}), 400
            
        report_data = {
            "reporter_id": reporter_id,
            "reported_id": reported_id,
            "job_id": job_id,
            "reason": reason,
            "status": "pending"
        }
        res = supabase.table('reports').insert(report_data).execute()
        return jsonify({"data": res.data[0], "message": "Report submitted successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/reports', methods=['GET'])
@require_auth
def get_reports():
    """Admin fetches all reports"""
    try:
        admin_id = request.user.id
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden"}), 403
            
        try:
            res = supabase.table('reports').select('*, jobs(title, company_name)').order('created_at', desc=True).execute()
            return jsonify({"data": res.data}), 200
        except Exception as join_err:
            print("Join error:", str(join_err))
            res = supabase.table('reports').select('*').order('created_at', desc=True).execute()
            return jsonify({"data": res.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/users', methods=['GET'])
@require_auth
def get_users():
    """Admin fetches all users with email from auth"""
    try:
        admin_id = request.user.id
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden"}), 403
            
        res = supabase.table('profiles').select('*').order('created_at', desc=True).execute()
        profiles = res.data or []
        
        # Try to attach emails using service role client if possible
        try:
            # Re-initialize the client with service role key to access auth admin
            import os
            from supabase import create_client
            url = os.environ.get("SUPABASE_URL")
            service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
            
            if url and service_key:
                admin_supabase = create_client(url, service_key)
                auth_users = admin_supabase.auth.admin.list_users()
                email_map = {u.id: u.email for u in auth_users}
                banned_map = {u.id: u.user_metadata.get('is_banned', False) if getattr(u, 'user_metadata', None) else False for u in auth_users}
                for p in profiles:
                    p['email'] = email_map.get(p['id'], p.get('email', 'No Email'))
                    p['is_banned'] = banned_map.get(p['id'], False)
        except Exception as e:
            print("Could not fetch auth emails or status:", e)
            
        # Aggregate counters for students and employers
        # Fetch jobs
        jobs_res = supabase.table('jobs').select('id, employer_id, status').execute()
        all_jobs = jobs_res.data or []
        
        # Fetch applications
        apps_res = supabase.table('applications').select('id, student_id, job_id, status').execute()
        all_apps = apps_res.data or []
        
        employer_jobs_map = {} # Count of active jobs for employers
        employer_hires_map = {} # Count of total hires for employers
        student_jobs_map = {} # Count of completed jobs for students
        
        # 1. Map Jobs to Employers and count active ones
        job_to_employer = {}
        for job in all_jobs:
            job_id = job.get('id')
            eid = job.get('employer_id')
            if eid:
                job_to_employer[job_id] = eid
                if job.get('status') == 'open':
                    employer_jobs_map[eid] = employer_jobs_map.get(eid, 0) + 1
                    
        # 2. Map Applications to count Hires
        for app in all_apps:
            sid = app.get('student_id')
            job_id = app.get('job_id')
            status = app.get('status')
            
            # If an application is accepted or completed, it counts as a hire for the employer
            if status in ['accepted', 'completed']:
                eid = job_to_employer.get(job_id)
                if eid:
                    employer_hires_map[eid] = employer_hires_map.get(eid, 0) + 1
                    
        # 3. Map Completed Shifts for Student Completed Jobs count
        shifts_res = supabase.table('shifts').select('student_id, status').execute()
        student_shifts = shifts_res.data or []
        for shift in student_shifts:
            if shift.get('status') == 'Completed' and shift.get('student_id'):
                sid = shift['student_id']
                student_jobs_map[sid] = student_jobs_map.get(sid, 0) + 1

        
        # Fetch reviews to compute average ratings for students
        reviews_res = supabase.table('reviews').select('student_id, rating').execute()
        student_reviews = reviews_res.data or []
        student_ratings = {}
        student_review_counts = {}
        for rev in student_reviews:
            sid = rev.get('student_id')
            rat = rev.get('rating')
            if sid and rat is not None:
                student_ratings[sid] = student_ratings.get(sid, 0) + rat
                student_review_counts[sid] = student_review_counts.get(sid, 0) + 1

        for p in profiles:
            p_id = p['id']
            if p.get('role') == 'student':
                p['completedJobs'] = student_jobs_map.get(p_id, 0)
                if student_review_counts.get(p_id, 0) > 0:
                    p['rating'] = round(student_ratings[p_id] / student_review_counts[p_id], 1)
                else:
                    p['rating'] = 0
            elif p.get('role') == 'employer':
                p['activeJobs'] = employer_jobs_map.get(p_id, 0)
                p['totalHires'] = employer_hires_map.get(p_id, 0)

        employer_ids = [p['id'] for p in profiles if p.get('role') == 'employer']
        if employer_ids:
            verifs = supabase.table('employer_verifications').select('employer_id, status').in_('employer_id', employer_ids).execute()
            verif_map = {v['employer_id']: v['status'] for v in verifs.data}
            for p in profiles:
                if p.get('role') == 'employer':
                    p['verification_status'] = verif_map.get(p['id'], 'approved' if p.get('is_verified') else 'pending')

        return jsonify({"data": profiles}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/jobs', methods=['GET'])
@require_auth
def get_all_jobs():
    """Admin fetches all jobs"""
    try:
        admin_id = request.user.id
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden"}), 403
            
        try:
            res = supabase.table('jobs').select('*, profiles!jobs_employer_id_fkey(full_name, company_name)').order('created_at', desc=True).execute()
            return jsonify({"data": res.data}), 200
        except Exception as join_err:
            print("Join error in admin jobs:", str(join_err))
            res = supabase.table('jobs').select('*').order('created_at', desc=True).execute()
            return jsonify({"data": res.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/jobs/<job_id>', methods=['DELETE'])
@require_auth
def admin_delete_job(job_id):
    """Admin deletes a job"""
    try:
        admin_id = request.user.id
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden"}), 403
            
        # First delete existing applications assigned to this job
        supabase.table('applications').delete().eq('job_id', job_id).execute()
        
        res = supabase.table('jobs').delete().eq('id', job_id).execute()
        return jsonify({"message": "Job deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/users/<user_id>/ban', methods=['POST'])
@require_auth
def toggle_ban_user(user_id):
    """Admin bans or unbans a user natively via Supabase Auth"""
    import os
    import requests
    try:
        admin_id = request.user.id
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden"}), 403
            
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
        headers = {'apikey': key, 'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        
        # Check current status
        res = requests.get(f"{url}/auth/v1/admin/users/{user_id}", headers=headers)
        if res.status_code != 200:
            return jsonify({"error": "User auth not found"}), 404
            
        auth_user = res.json()
        is_currently_banned = bool(auth_user.get('banned_until'))
        
        is_banning = not is_currently_banned
        payload = {
            "ban_duration": "87600h" if is_banning else "none",
            "user_metadata": {**auth_user.get('user_metadata', {}), "is_banned": is_banning}
        }
        
        update_res = requests.put(f"{url}/auth/v1/admin/users/{user_id}", headers=headers, json=payload)
        
        # Sub-action: If the user is an employer and banned, close all their active jobs
        if is_banning:
            try:
                supabase.table('jobs').update({'status': 'closed'}).eq('employer_id', user_id).execute()
            except Exception as e:
                print("Failed to close jobs for banned user:", e)
            
        return jsonify({"message": f"User {'banned' if is_banning else 'unbanned'} successfully", "data": update_res.json()}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- ADMIN / VERIFICATIONS API ---

@app.route('/api/verifications', methods=['POST'])
@require_auth
def submit_verification():
    """Employer submits KYC documents for verification"""
    try:
        data = request.json
        employer_id = request.user.id
        
        # Verify user is an employer
        profile_res = supabase.table('profiles').select('role').eq('id', employer_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'employer':
            return jsonify({"error": "Forbidden", "message": "Only employers can submit verifications"}), 403
            
        verif_data = {
            "employer_id": employer_id,
            "business_name": data.get("business_name"),
            "registration_number": data.get("registration_number"),
            "document_url": data.get("document_url"),
            "status": "pending"
        }

        # Check if verification already exists
        existing = supabase.table('employer_verifications').select('id').eq('employer_id', employer_id).execute()
        if existing.data:
            # Update the existing record so they can resubmit
            response = supabase.table('employer_verifications').update(verif_data).eq('employer_id', employer_id).execute()
        else:
            # Insert a new record
            response = supabase.table('employer_verifications').insert(verif_data).execute()
            
        return jsonify({"data": response.data[0] if response.data else {}}), 201
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/verifications', methods=['GET'])
@require_auth
def get_verifications():
    """Admin fetches all employer verifications"""
    try:
        admin_id = request.user.id
        
        # Verify user is an admin
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden", "message": "Only admins can view verifications"}), 403
            
        # Fetch verifications with employer details (profiles only has full_name)
        response = supabase.table('employer_verifications').select('*, profiles(full_name)').order('submitted_at', desc=True).execute()
        merchants = response.data
        
        # Merge email and phone from Supabase Auth Admin API
        try:
            import os
            from supabase import create_client
            url = os.environ.get("SUPABASE_URL")
            service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
            admin_supabase = create_client(url, service_key)
            
            auth_users = admin_supabase.auth.admin.list_users()
            email_map = {u.id: u.email for u in auth_users}
            phone_map = {}
            for u in auth_users:
                phone = u.phone
                if not phone and u.user_metadata:
                    phone = u.user_metadata.get('phone')
                phone_map[u.id] = phone
            
            for m in merchants:
                if m.get('profiles') and m.get('employer_id'):
                    eid = m['employer_id']
                    m['profiles']['email'] = email_map.get(eid, 'No Email')
                    m['profiles']['phone'] = phone_map.get(eid, 'No Phone')
        except Exception as auth_err:
            print("Could not fetch auth details:", auth_err)

        return jsonify({"data": merchants}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/verifications/approve', methods=['POST'])
@require_auth
def approve_verification():
    """Admin approves or rejects an employer verification"""
    try:
        data = request.json
        admin_id = request.user.id
        verification_id = data.get('verification_id')
        employer_id = data.get('employer_id')
        status = data.get('status') # 'approved' or 'rejected'
        notes = data.get('notes', '')
        
        if not verification_id or not employer_id or not status:
            return jsonify({"error": "Missing required fields"}), 400
            
        # Verify user is an admin
        profile_res = supabase.table('profiles').select('role').eq('id', admin_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'admin':
            return jsonify({"error": "Forbidden", "message": "Only admins can process verifications"}), 403
            
        # Update verification record
        update_data = {
            "status": status,
            "reviewed_by": admin_id,
            "reviewed_at": datetime.utcnow().isoformat(),
            "notes": notes
        }
        verif_res = supabase.table('employer_verifications').update(update_data).eq('id', verification_id).execute()
        
        # If approved, update employer profile
        if status == 'approved':
            supabase.table('profiles').update({'is_verified': True}).eq('id', employer_id).execute()
            
        return jsonify({"message": f"Verification {status} successfully", "data": verif_res.data[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
