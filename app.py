import os
import requests
import math
import io
import base64
import json
import time
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, request
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import PIL.Image
# google-genai SDK no longer needed — using direct REST API for Gemini

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for the React frontend and restrict to allowed origins
ALLOWED_ORIGINS = [
    os.environ.get("FRONTEND_URL", "https://job-finder-frontend-lake.vercel.app"),
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
]
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# In-memory rate limiter for sensitive endpoints (IP -> list of timestamps)
_rate_limit_store: dict[str, list[float]] = {}
RATE_LIMIT_MAX = 5       # max requests
RATE_LIMIT_WINDOW = 60   # per 60 seconds

def _is_rate_limited(ip: str) -> bool:
    """Return True if the IP has exceeded RATE_LIMIT_MAX requests in the last RATE_LIMIT_WINDOW seconds."""
    now = time.time()
    timestamps = _rate_limit_store.get(ip, [])
    # Prune old entries
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(timestamps) >= RATE_LIMIT_MAX:
        _rate_limit_store[ip] = timestamps
        return True
    timestamps.append(now)
    _rate_limit_store[ip] = timestamps
    return False

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Lightweight endpoint for keep-alive pings. Prevents Render cold starts."""
    return jsonify({"status": "ok"}), 200

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

        # Handle data URL vs raw base64
        if image_data.startswith('data:'):
            mime_type = image_data.split(';')[0].split(':')[1]
            base64_img = image_data.split(',')[1]
        else:
            mime_type = "image/jpeg"
            base64_img = image_data

        prompt = """Extract the following information from this job poster image.
Return ONLY a raw JSON object with these exact keys:
- title: The job title (e.g. "Cashier", "Tutor")
- location: The place, city, or address mentioned (e.g. "Kaliyakkavilai"). If not found, use "".
- description: A brief summary of the role. If not clearly stated, combine other details like Age limit, Qualifications, and contact info into a readable description.
- wage: The daily pay rate, just the number (e.g. "500"). If the poster lists a monthly income (e.g. "15000 - 20000"), divide the lower number by 30 to approximate the daily wage and return only that number. If not found, use "".
- requirements: Any shift timings or requirements mentioned (e.g. "9.30 am - 4.30 pm").
Do not include markdown formatting or json code blocks, just raw JSON."""

        # --- Try Groq API first (free, fast, generous limits) ---
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            print("Using Groq API for image parsing...")
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            groq_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}"
            }
            groq_payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }

            resp = requests.post(groq_url, headers=groq_headers, json=groq_payload, timeout=60)

            if resp.status_code == 200:
                result = resp.json()
                text = result['choices'][0]['message']['content']
                text = text.replace('```json', '').replace('```', '').strip()
                parsed_data = json.loads(text)
                return jsonify({"data": parsed_data}), 200
            else:
                print(f"Groq API failed ({resp.status_code}): {resp.text}")
                # Fall through to Gemini

        # --- Fallback: Gemini API ---
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            return jsonify({"error": "No AI API key configured on server"}), 500

        print("Using Gemini API for image parsing...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_img
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)

        if resp.status_code != 200:
            err_detail = resp.text
            try:
                err_detail = resp.json()
            except:
                pass
            print(f"Gemini API Error: {err_detail}")
            return jsonify({"error": f"AI Parsing Error ({resp.status_code})"}), 500

        result = resp.json()
        try:
            text = result['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            print(f"Unexpected Gemini response format: {result}")
            return jsonify({"error": "Unexpected response format from AI"}), 500

        text = text.replace('```json', '').replace('```', '').strip()
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
            "vacancies": data.get("vacancies", 1),
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
        job_res = supabase.table('jobs').select('employer_id, status, vacancies').eq('id', job_id).execute()
        if not job_res.data or job_res.data[0]['employer_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403
            
        allowed_fields = ['title', 'description', 'location', 'wage', 'requirements', 'urgent', 'status', 'image_url', 'vacancies', 'pause_reason']
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        # Auto-clear pause_reason when re-opening a job
        if update_data.get('status') == 'open':
            update_data['pause_reason'] = None
        
        # If vacancies increased on a closed job, check if we should auto-reopen
        current_status = job_res.data[0].get('status', 'open') if 'status' not in update_data else update_data['status']
        if 'vacancies' in update_data and current_status == 'closed':
            # Count currently accepted applications
            accepted_res = supabase.table('applications').select('id', count='exact').eq('job_id', job_id).eq('status', 'accepted').execute()
            current_accepted = accepted_res.count if hasattr(accepted_res, 'count') and accepted_res.count is not None else len(accepted_res.data)
            new_vacancies = int(update_data['vacancies'])
            if new_vacancies > current_accepted:
                update_data['status'] = 'open'
                update_data['pause_reason'] = None
        
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

@app.route('/api/messages/<message_id>', methods=['DELETE'])
@require_auth
def delete_message(message_id):
    """Delete a message — only the sender can delete their own message"""
    try:
        user_id = request.user.id
        # Verify the message belongs to the requesting user
        msg_res = supabase.table('messages').select('sender_id').eq('id', message_id).execute()
        if not msg_res.data:
            return jsonify({"error": "Message not found"}), 404
        if msg_res.data[0]['sender_id'] != user_id:
            return jsonify({"error": "Forbidden – you can only delete your own messages"}), 403
        supabase.table('messages').delete().eq('id', message_id).execute()
        return jsonify({"message": "Message deleted"}), 200
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
        # Verify ownership / access
        shift_res = supabase.table('shifts').select('employer_id, student_id').eq('id', shift_id).execute()
        if not shift_res.data:
            return jsonify({'error': 'Shift not found'}), 404
        shift = shift_res.data[0]
        if user_id != shift.get('employer_id') and user_id != shift.get('student_id'):
            return jsonify({'error': 'Forbidden - you do not have access to this shift'}), 403

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
        user_id = request.user.id
        # Verify ownership
        shift_res = supabase.table('shifts').select('employer_id').eq('id', shift_id).execute()
        if not shift_res.data:
            return jsonify({'error': 'Shift not found'}), 404
        if user_id != shift_res.data[0].get('employer_id'):
            return jsonify({'error': 'Forbidden - only the shift creator can delete it'}), 403

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
            
        # Verify job is still open
        job_check = supabase.table('jobs').select('status').eq('id', job_id).execute()
        if not job_check.data or job_check.data[0]['status'] != 'open':
            return jsonify({"error": "Forbidden", "message": "This job is no longer accepting applications"}), 400

        # Verify no duplicate application
        existing = supabase.table('applications').select('id').eq('job_id', job_id).eq('student_id', student_id).execute()
        if existing.data:
            return jsonify({"error": "Conflict", "message": "You have already applied to this job"}), 409
            
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

@app.route('/api/auth/check-email', methods=['POST'])
def check_email_exists():
    """Rate-limited check whether an email is registered (for OTP login UX)."""
    # --- Rate limit by IP ---
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr) or 'unknown'
    if _is_rate_limited(client_ip):
        return jsonify({"error": "Too many requests. Please try again later."}), 429

    try:
        data = request.json
        email = (data.get('email') or '').strip().lower()
        if not email:
            return jsonify({"error": "Email is required"}), 400

        service_role_key = os.getenv('SUPABASE_KEY')
        supabase_url = os.getenv('SUPABASE_URL')

        headers = {
            'apikey': service_role_key,
            'Authorization': f'Bearer {service_role_key}'
        }

        # Query Supabase Admin API for specific email
        users_res = requests.get(
            f'{supabase_url}/auth/v1/admin/users',
            headers=headers,
            params={'page': 1, 'per_page': 1000}
        )
        if users_res.status_code == 200:
            for u in users_res.json().get('users', []):
                if (u.get('email') or '').lower() == email:
                    return jsonify({"exists": True}), 200

        return jsonify({"exists": False}), 200

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
        apps = response.data or []
        
        if apps:
            job_ids = [a['job_id'] for a in apps]
            shifts_res = supabase.table('shifts').select('*').eq('student_id', student_id).in_('job_id', job_ids).in_('status', ['Pending', 'Confirmed']).execute()
            shifts_data = shifts_res.data or []
            shifts_map = {s['job_id']: s for s in shifts_data}
            
            for app in apps:
                shift = shifts_map.get(app['job_id'])
                if shift:
                    app['shift_id'] = shift['id']
                    if app['status'] == 'pending' and shift['status'] == 'Pending':
                        app['status'] = 'interview'

        return jsonify({"data": apps}), 200
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
        
        # Step 5: Fetch shifts to dynamically set interview status
        shifts_res = supabase.table('shifts').select('*').in_('job_id', job_ids).in_('status', ['Pending', 'Confirmed']).execute()
        shifts_data = shifts_res.data or []
        shifts_map = {(s['job_id'], s['student_id']): s for s in shifts_data}
        
        enriched = []
        for app in apps:
            app['jobs'] = jobs_map.get(app['job_id'], {})
            app['profiles'] = profiles_map.get(app['student_id'], {})
            
            shift = shifts_map.get((app['job_id'], app['student_id']))
            if shift:
                app['shift_id'] = shift['id']
                if app['status'] == 'pending' and shift['status'] == 'Pending':
                    app['status'] = 'interview'
            
            enriched.append(app)
        
        print(f"[EMPLOYER_APPS] Returning {len(enriched)} enriched applications")
        return jsonify({"data": enriched}), 200
    except Exception as e:
        import traceback
        print(f"[EMPLOYER_APPS] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/employer/analytics', methods=['GET'])
@require_auth
def get_employer_analytics():
    """Returns application metrics for the last 7 days for the employer's jobs."""
    try:
        employer_id = request.user.id
        
        # Verify user is an employer
        profile_res = supabase.table('profiles').select('role').eq('id', employer_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'employer':
            return jsonify({"error": "Forbidden", "message": "Only employers can view analytics"}), 403
            
        # Get all job IDs that belong to this employer
        jobs_res = supabase.table('jobs').select('id').eq('employer_id', employer_id).execute()
        if not jobs_res.data:
            return jsonify({"data": []}), 200
        job_ids = [j['id'] for j in jobs_res.data]
        
        # Calculate the date 7 days ago
        from datetime import datetime, timedelta
        seven_days_ago = (datetime.utcnow() - timedelta(days=6)).strftime('%Y-%m-%d')
        
        # Fetch applications for those jobs created in the last 7 days
        app_res = supabase.table('applications')\
            .select('created_at')\
            .in_('job_id', job_ids)\
            .gte('created_at', seven_days_ago)\
            .execute()
            
        apps = app_res.data or []
        
        import random
        
        # Build the last 7 days array
        days = []
        for i in range(6, -1, -1):
            d = datetime.now() - timedelta(days=i)
            day_name = d.strftime('%a') # Mon, Tue, etc
            date_str = d.strftime('%Y-%m-%d')
            days.append({'date': date_str, 'name': day_name, 'applications': 0, 'views': 0})
            
        # Count applications per day
        for app in apps:
            app_date = app['created_at'][:10]
            for day in days:
                if day['date'] == app_date:
                    day['applications'] += 1
                    break
                    
        # Simulate views (random 2.5x to 4x of applications + a small baseline)
        for day in days:
            base_views = random.randint(15, 60)
            day['views'] = int(day['applications'] * random.uniform(2.5, 4.0)) + base_views
            
        return jsonify({"data": days}), 200
    except Exception as e:
        import traceback
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
        app_res = supabase.table('applications').select('*, jobs!inner(employer_id, title, vacancies)').eq('id', application_id).execute()
        
        if not app_res.data or len(app_res.data) == 0:
             return jsonify({"error": "Not Found", "message": "Application not found"}), 404
             
        if app_res.data[0]['jobs']['employer_id'] != employer_id:
            return jsonify({"error": "Forbidden", "message": "You do not own this job"}), 403
            
        vacancies = app_res.data[0]['jobs'].get('vacancies', 1)
        job_id = app_res.data[0]['job_id']
        job_closed = False

        if new_status == 'accepted':
            # Check current accepted count
            accepted_res = supabase.table('applications').select('id', count='exact').eq('job_id', job_id).eq('status', 'accepted').execute()
            current_accepted = accepted_res.count if hasattr(accepted_res, 'count') and accepted_res.count is not None else len(accepted_res.data)
            
            if current_accepted >= vacancies:
                return jsonify({"error": "Vacancy Limit Reached", "message": "This job has already reached its maximum number of hires.", "auto_closed": True}), 400

        # Update the status
        update_res = supabase.table('applications').update({'status': new_status}).eq('id', application_id).execute()
        
        if new_status == 'accepted':
            if current_accepted + 1 >= vacancies:
                # Auto close the job to prevent more applicants!
                supabase.table('jobs').update({'status': 'closed', 'pause_reason': 'All vacancies for this role have been filled.'}).eq('id', job_id).execute()
                job_closed = True
        
        student_id = app_res.data[0]['student_id']
        job_title = app_res.data[0]['jobs'].get('title', 'a job')
        create_notification(student_id, f"Your application for {job_title} was {new_status}", "status_update")
        
        return jsonify({
            "data": update_res.data[0], 
            "message": f"Application {new_status} successfully",
            "auto_closed": job_closed
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- INTERVIEW SCHEDULER API ---

@app.route('/api/applications/<application_id>/schedule_interview', methods=['PUT'])
@require_auth
def schedule_interview(application_id):
    """Employer schedules an interview for an applicant.
    Stores interview info as a Pending shift in the shifts table."""
    try:
        employer_id = request.user.id
        data = request.json
        interview_date = data.get('interview_date')
        interview_time = data.get('interview_time')
        interview_link = data.get('interview_link', '')
        interview_notes = data.get('interview_notes', '')

        if not interview_date or not interview_time:
            return jsonify({"error": "Date and time are required"}), 400

        # Verify the application belongs to a job owned by this employer
        app_res = supabase.table('applications').select('*, jobs!inner(employer_id, title, id)').eq('id', application_id).execute()
        if not app_res.data:
            return jsonify({"error": "Application not found"}), 404
        if app_res.data[0]['jobs']['employer_id'] != employer_id:
            return jsonify({"error": "Forbidden", "message": "You do not own this job"}), 403

        student_id = app_res.data[0]['student_id']
        job_title = app_res.data[0]['jobs'].get('title', 'a job')
        job_id = app_res.data[0]['jobs'].get('id')

        # Calculate end time (1 hour after start)
        end_time_str = interview_time
        try:
            from datetime import timedelta
            t = datetime.strptime(interview_time, "%H:%M")
            end_t = t + timedelta(hours=1)
            end_time_str = end_t.strftime("%H:%M")
        except:
            pass
            
        # We do NOT update the DB status to 'interview' due to postgres constraints limit. 
        # The GET endpoints dynamically return 'interview' status when a shift exists.

        # Create a Pending shift for the interview
        notes_str = f"📅 Interview for {job_title}"
        if interview_link:
            notes_str += f" | Link: {interview_link}"
        if interview_notes:
            notes_str += f" | {interview_notes}"

        shift_data = {
            'employer_id': employer_id,
            'job_id': job_id,
            'student_id': student_id,
            'shift_date': interview_date,
            'start_time': interview_time,
            'end_time': end_time_str,
            'status': 'Pending',
            'notes': notes_str
        }
        shift_res = supabase.table('shifts').insert(shift_data).execute()

        # Format a human-readable date for the notification
        try:
            d_obj = datetime.strptime(interview_date, "%Y-%m-%d")
            formatted_date = d_obj.strftime("%B %d, %Y")
        except:
            formatted_date = interview_date

        # Notify the student
        create_notification(
            student_id,
            f"📅 Interview scheduled for {job_title} on {formatted_date} at {interview_time}. Check your Schedule to accept!",
            "interview"
        )

        # Send a special interview message in chat
        interview_details = {
            "interview_date": interview_date,
            "interview_time": interview_time,
            "interview_link": interview_link,
            "interview_notes": interview_notes,
            "job_title": job_title,
            "application_id": application_id,
            "shift_id": shift_res.data[0]['id'] if shift_res.data else None
        }
        interview_msg = f"[INTERVIEW_PROPOSAL]{json.dumps(interview_details)}"
        try:
            supabase.table('messages').insert({
                'sender_id': employer_id,
                'receiver_id': student_id,
                'content': interview_msg
            }).execute()
        except Exception as msg_err:
            print(f"Failed to send interview chat message: {msg_err}")

        return jsonify({"message": "Interview scheduled successfully", "data": shift_res.data[0] if shift_res.data else {}}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("error.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/applications/<application_id>/accept_interview', methods=['PUT'])
@require_auth
def accept_interview(application_id):
    """Student accepts an interview — confirms the pending shift on their calendar"""
    try:
        student_id = request.user.id

        # Verify the application belongs to this student
        app_res = supabase.table('applications').select('*, jobs!inner(employer_id, title, id)').eq('id', application_id).execute()
        if not app_res.data:
            return jsonify({"error": "Application not found"}), 404
        if app_res.data[0]['student_id'] != student_id:
            return jsonify({"error": "Forbidden"}), 403
        if app_res.data[0]['status'] != 'interview':
            return jsonify({"error": "No pending interview to accept"}), 400

        job_data = app_res.data[0]['jobs']
        employer_id = job_data['employer_id']
        job_title = job_data.get('title', 'Interview')
        job_id = job_data.get('id')

        # Update application status to accepted
        supabase.table('applications').update({'status': 'accepted'}).eq('id', application_id).execute()

        # Find and confirm the Pending shift for this interview
        pending_shifts = supabase.table('shifts').select('*').eq('student_id', student_id).eq('job_id', job_id).eq('status', 'Pending').order('created_at', desc=True).limit(1).execute()
        if pending_shifts.data:
            supabase.table('shifts').update({'status': 'Confirmed'}).eq('id', pending_shifts.data[0]['id']).execute()

        # Notify the employer
        student_res = supabase.table('profiles').select('full_name').eq('id', student_id).execute()
        student_name = student_res.data[0]['full_name'] if student_res.data else "A student"
        create_notification(
            employer_id,
            f"✅ {student_name} accepted the interview for {job_title}!",
            "interview"
        )

        return jsonify({"message": "Interview accepted! It has been added to your schedule."}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
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

# --- AI DOCUMENT VERIFICATION HELPERS ---

def validate_id_format(id_number):
    """Validate Indian ID number formats (Aadhaar, PAN, GSTIN)"""
    import re
    result = {"type": "unknown", "valid": False, "details": ""}
    
    if not id_number:
        return result
    
    clean = id_number.strip().upper().replace(" ", "")
    
    # PAN: ABCDE1234F
    if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', clean):
        result["type"] = "PAN"
        result["valid"] = True
        result["details"] = f"Valid PAN format: {clean}"
        return result
    
    # GSTIN: 15 chars, starts with 2-digit state code
    if re.match(r'^[0-1][1-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$|^[2-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$', clean):
        result["type"] = "GSTIN"
        result["valid"] = True
        result["details"] = f"Valid GSTIN format: {clean}"
        return result
    
    # Aadhaar: 12 digits with Verhoeff checksum
    if re.match(r'^[0-9]{12}$', clean):
        result["type"] = "Aadhaar"
        # Verhoeff checksum validation
        verhoeff_table_d = [
            [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],
            [2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
            [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
            [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
            [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]
        ]
        verhoeff_table_p = [
            [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],
            [5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
            [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
            [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]
        ]
        
        try:
            c = 0
            for i, digit in enumerate(reversed(clean)):
                c = verhoeff_table_d[c][verhoeff_table_p[i % 8][int(digit)]]
            result["valid"] = (c == 0)
            result["details"] = f"Aadhaar format {'valid' if result['valid'] else 'INVALID checksum'}: {clean[:4]}****{clean[8:]}"
        except:
            result["valid"] = False
            result["details"] = "Could not validate Aadhaar checksum"
        return result
    
    # Strict generic numeric registration (only if exactly 6-20 length, with a healthy mix of letters/numbers usually, not just random letters)
    # We will remove the generic fallback that caught ASDFGH123456 as valid.
    # Now, if it isn't PAN, GSTIN, or Aadhaar, we just return it as unvalidated generic.
    
    result["details"] = f"Unrecognized/Generic ID format: {clean}"
    # Still return valid=True so it doesn't instantly penalize them if they uploaded a legit local shop license that doesn't follow federal formats
    result["valid"] = True 
    result["type"] = "Generic ID"
    return result


def analyze_document_with_ai(document_url, business_name, registration_number):
    """Analyze an employer's verification document using Groq Llama 4 Vision"""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return {"error": "GROQ_API_KEY not configured", "confidence": 0}
        
        # Ensure proper data URL
        image_data = document_url
        if not image_data.startswith('data:'):
            image_data = f"data:image/jpeg;base64,{image_data}"
        
        prompt = f"""You are a KYC (Know Your Customer) verification AI for an employment platform in India.
Analyze this business verification document carefully.

Reference Data provided by user:
- Claimed Name: "{business_name}"
- Claimed ID: "{registration_number}"

Examine the document image and return ONLY a raw JSON object with these exact keys:
- document_type: What type of document is this? (e.g. "Aadhaar Card", "PAN Card", "GSTIN Certificate", "Shop License", "Business Registration", "Unknown")
- extracted_name: Extract the name exactly as it appears on the document. Look hard for it. If absolutely unreadable, use "".
- extracted_id: Extract the ID/registration number exactly as it appears. If absolutely unreadable, use "".
- is_authentic: true or false - does this look like a genuine document?
- confidence: A number from 0 to 100 indicating your confidence in the document's authenticity.
- red_flags: An array of strings listing genuine concerns (e.g. "Blurry text", "Possible editing detected", "Missing official seal", "Expired document"). Do NOT flag "Missing photo" or "Blank photo area" UNLESS the document type strictly requires a photo (like Aadhaar/DL). For text-only certificates, a missing photo is NORMAL. 
- summary: A one-sentence assessment of the document.

Do not include markdown formatting or json code blocks, just raw JSON."""

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if resp.status_code != 200:
            return {"error": f"AI request failed: {resp.status_code}", "confidence": 0}
        
        result = resp.json()
        text = result['choices'][0]['message']['content']
        text = text.replace('```json', '').replace('```', '').strip()
        ai_report = json.loads(text)
        
        # Layer 2: Format validation on the registration number
        id_validation = validate_id_format(registration_number)
        ai_report["id_format_check"] = id_validation
        
        # Handle missing Extractions gracefully - Use Claimed data as fallback if AI entirely fails
        ai_extracted_name = ai_report.get("extracted_name", "")
        ai_extracted_id = ai_report.get("extracted_id", "")
        
        if not ai_extracted_name or str(ai_extracted_name).strip().lower() in ["", "none", "not found", "null"]:
            ai_report["extracted_name"] = business_name
            # Slightly deduce confidence for needing a fallback
            ai_report["confidence"] = max(0, ai_report.get("confidence", 50) - 5)
            
        if not ai_extracted_id or str(ai_extracted_id).strip().lower() in ["", "none", "not found", "null"]:
            ai_report["extracted_id"] = registration_number
            # Slightly deduce confidence for needing a fallback
            ai_report["confidence"] = max(0, ai_report.get("confidence", 50) - 5)

        # Layer 3: Cross-reference names
        extracted = str(ai_report.get("extracted_name")).strip().lower()
        claimed = str(business_name).strip().lower()
        if extracted and claimed:
            # Simple fuzzy match: check if one contains the other
            name_match = extracted in claimed or claimed in extracted or extracted == claimed
            ai_report["name_match"] = name_match
            if not name_match:
                ai_report.setdefault("red_flags", []).append(
                    f"Name mismatch: document shows '{ai_report.get('extracted_name')}' but account registered as '{business_name}'"
                )
                # Reduce confidence if names don't match
                ai_report["confidence"] = max(0, ai_report.get("confidence", 50) - 20)
        else:
            ai_report["name_match"] = None
        
        # Reduce confidence if ID format is invalid
        if not id_validation["valid"] and id_validation["type"] != "unknown":
            ai_report.setdefault("red_flags", []).append(f"Invalid {id_validation['type']} format")
            ai_report["confidence"] = max(0, ai_report.get("confidence", 50) - 15)
        
        # Final recommendation
        confidence = ai_report.get("confidence", 0)
        red_flags = ai_report.get("red_flags", [])
        if confidence >= 85 and len(red_flags) == 0:
            ai_report["recommendation"] = "auto_approve"
        elif confidence >= 50:
            ai_report["recommendation"] = "manual_review"
        else:
            ai_report["recommendation"] = "likely_reject"
        
        return ai_report
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "confidence": 0, "recommendation": "manual_review"}


# --- ADMIN / VERIFICATIONS API ---

@app.route('/api/verifications', methods=['POST'])
@require_auth
def submit_verification():
    """Employer submits KYC documents for verification — AI auto-analyzes"""
    try:
        data = request.json
        employer_id = request.user.id
        
        # Verify user is an employer
        profile_res = supabase.table('profiles').select('role').eq('id', employer_id).execute()
        if not profile_res.data or profile_res.data[0]['role'] != 'employer':
            return jsonify({"error": "Forbidden", "message": "Only employers can submit verifications"}), 403
            
        business_name = data.get("business_name")
        registration_number = data.get("registration_number")
        document_url = data.get("document_url")
        # Check if verification already exists to get previous attempts
        existing = supabase.table('employer_verifications').select('id, ai_analysis').eq('employer_id', employer_id).execute()
        
        attempts = 1
        if existing.data and existing.data[0].get('ai_analysis'):
            try:
                old_stats = existing.data[0]['ai_analysis']
                if isinstance(old_stats, str):
                    old_stats = json.loads(old_stats)
                attempts = old_stats.get("submission_attempts", 0) + 1
            except:
                pass

        if attempts > 3:
            return jsonify({"error": "Max attempts reached", "message": "You have reached the maximum number of verification attempts. Please contact support."}), 403

        # Run AI analysis on the document
        ai_report = {}
        if document_url:
            ai_report = analyze_document_with_ai(document_url, business_name, registration_number)
            
        # Add tracking data to AI report
        ai_report["submission_attempts"] = attempts
        
        # Determine initial status based on AI confidence
        status = "pending"
        if ai_report.get("recommendation") == "auto_approve":
            status = "approved"
        
        verif_data = {
            "employer_id": employer_id,
            "business_name": business_name,
            "registration_number": registration_number,
            "document_url": document_url,
            "status": status,
            "ai_analysis": json.dumps(ai_report) if ai_report else json.dumps({"submission_attempts": attempts})
        }

        if existing.data:
            # Update the existing record so they can resubmit
            response = supabase.table('employer_verifications').update(verif_data).eq('employer_id', employer_id).execute()
        else:
            # Insert a new record
            response = supabase.table('employer_verifications').insert(verif_data).execute()
        
        # If auto-approved, also update the employer profile
        if status == "approved":
            supabase.table('profiles').update({'is_verified': True}).eq('id', employer_id).execute()
            
        return jsonify({
            "data": response.data[0] if response.data else {}, 
            "ai_analysis": ai_report,
            "attempts_remaining": 3 - attempts
        }), 201
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
