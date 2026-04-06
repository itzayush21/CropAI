from flask import Flask,render_template,request,redirect, session, flash, url_for, jsonify
from config import Config
from models import db
from auth.auth_client import create_supabase_client
from module.preprocessing import enrich_user_data,get_lat_lon_from_location
from sqlalchemy.orm.attributes import flag_modified
from models import db, User, SoilDB, WeatherDB, CropSuggestionDB, FertilizerSuggestionDB, PestControlDB, FinancialAdvisorDB, CropRoom, NearbyService, InventoryStore
from module.pest_control_genai import generate_pest_control_advice
from module.genai_crop_advisor import generate_crop_suggestions
from module.fertilizer_genai import generate_fertilizer_recommendations
from module.financial_genai import generate_financial_advice
from module.crop_suggestion_ai import generate_crop_suggestion
from module.gemini_crop_pipeline import generate_initial_step, generate_next_step, format_step_for_db
from module.gemini_advisors import get_fertilizer_suggestions, get_pest_guidelines, get_financial_advice, get_ai_doubt_response
from module.disease_pipeline import run_yolo_classification, get_gemini_guidance
from module.ai_explainer import run_ai_explainer
from module.decision_engine import run_decision_engine
from module.nearby_services_engine import find_nearby_services
from module.equipment_module import generate_equipment_recommendations
from module.invent_summary import generate_inventory_summary
import time
import uuid
import math
import json
import tempfile
from datetime import datetime, timezone


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

supabase = create_supabase_client()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        district = request.form.get('district')
        state = request.form.get('state')
        address = request.form.get('address')
        land_area = request.form.get('land_area')
        soil_summary = request.form.get('soil_summary')
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')

        try:
            res = supabase.auth.sign_up({"email": email, "password": password})
            print("[DEBUG] Supabase signup response:", res)
        except Exception as e:
            print("[ERROR] Supabase signup failed:", e)
            flash(f"Signup failed: {e}", "error")
            return render_template("register.html")

        if not res.user:
            print("[ERROR] Supabase signup failed:", res)
            flash("Registration failed. Try again later.", "error")
            return render_template("register.html")

        # Run preprocessing dynamically
        try:
            enriched = enrich_user_data(
                address=address,
                district=district,
                state=state,
                latitude=lat if lat else None,
                longitude=lon if lon else None
            )
        except Exception as e:
            flash(f"Data enrichment failed: {e}", "error")
            enriched = {"latitude": lat, "longitude": lon, "ndvi": None, "ndwi": None, "land_summary": None}

        new_user = User(
            userid=res.user.id,
            name=name,
            latitude=enriched["latitude"],
            longitude=enriched["longitude"],
            land_area=land_area,
            ndvi=enriched["ndvi"],
            ndwi=enriched["ndwi"],
            land_summary=enriched["land_summary"],
            soil_summary=soil_summary,
            district=district,
            state=state
        )

        db.session.add(new_user)
        db.session.commit()

        session['user'] = {"id": res.user.id, "email": email, "name": name}
        flash("Registration successful! 🌱", "success")
        return redirect('/dashboard')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        except Exception as e:
            flash("Error connecting to authentication service.", "error")
            return render_template("login.html")

        if res.user:
            # Set session variables
            session['user'] = {
                "id": res.user.id,
                "email": res.user.email
            }
            session['access_token'] = res.session.access_token

            # Check if user exists in local DB
            existing_user = User.query.filter_by(userid=res.user.id).first()

            if existing_user:
                flash("Login successful!", "success")
                return redirect('/dashboard')
            else:
                flash("User not found in local database. Please complete your profile.", "warning")
                return redirect('/register')

        else:
            flash("Invalid credentials. Please try again.", "error")

    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    # 1️⃣ Get the logged-in user's ID from session
    user_id = session['user']['id']

    # 2️⃣ Fetch the user's name from User table
    user = User.query.filter_by(userid=user_id).first()
    if not user:
        return redirect(url_for('login'))

    # 3️⃣ Use that name to fetch crop rooms
    crop_rooms = CropRoom.query.filter_by(username=user.name).all()

    print(f"[DEBUG] Crop Rooms for {user.name}: {len(crop_rooms)} found")

    # 4️⃣ Render dashboard (landing.html)
    return render_template("landing.html", crop_rooms=crop_rooms, user=user)




# MODULE

@app.route('/module/crop-suggestor')
def crop_suggestor_module():
    if 'user' not in session:
        return redirect(url_for('login'))
    # Simply render the crop advisor chat interface
    return render_template('crop_advisor.html')



# Page
@app.route('/crop-advisor', methods=['GET'])
def crop_advisor_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('crop_advisor.html')

# API: build context → call GenAI → persist → return JSON
@app.route('/api/crop-advisor', methods=['POST'])
def crop_advisor_api():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(force=True) or {}
    overrides = payload.get("context_overrides", {}) or {}
    user_query = payload.get("user_query")

    # 1) Fetch user + soil + weather context
    user = User.query.filter_by(userid=session['user']['id']).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    soil = SoilDB.query.filter_by(district=user.district, state=user.state).first()
    weather = WeatherDB.query.filter_by(district=user.district, state=user.state).first()

    # 2) Build context JSON (also storeable/loggable if needed)
    context_json = {
        "user": {
            "name": user.name,
            "district": user.district,
            "state": user.state,
            "land_area_acres": overrides.get("land_area_override") or user.land_area,
            "ndvi": user.ndvi,
            "ndwi": user.ndwi,
            "soil_summary": user.soil_summary,
            "land_summary": user.land_summary,
        },
        "soil": {
            "soil_type": getattr(soil, "soil_type", None),
            "ph_level": getattr(soil, "ph_level", None),
            "organic_carbon": getattr(soil, "organic_carbon", None),
            "nitrogen": getattr(soil, "nitrogen", None),
            "phosphorus": getattr(soil, "phosphorus", None),
            "potassium": getattr(soil, "potassium", None),
        } if soil else None,
        "weather": {
            "avg_temp": getattr(weather, "avg_temp", None),
            "rainfall": getattr(weather, "rainfall", None),
            "humidity": getattr(weather, "humidity", None),
            "summary": getattr(weather, "weather_summary", None),
        } if weather else None,
        "overrides": {
            "season": overrides.get("season"),
            "irrigation": overrides.get("irrigation"),
            "budget_inr": overrides.get("budget"),
            "market_pref": overrides.get("market_pref"),
            "risk": overrides.get("risk"),
            "notes": overrides.get("notes"),
        }
    }

    # 3) Call GenAI
    try:
        ai_result = generate_crop_suggestions(context_json=context_json, user_query=user_query)
        print("AI Result:", ai_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 4) Persist to CropSuggestionDB
    try:
        rec = CropSuggestionDB(
            district=user.district,
            state=user.state,
            suitable_crops=ai_result.get("suitable_crops", []),
            reasoning=ai_result.get("reasoning", "")
        )
        db.session.add(rec)
        db.session.commit()
    except Exception as e:
        # Non-fatal for frontend; still show suggestions
        db.session.rollback()

    # 5) Return to frontend (include district/state for UX)
    ai_result["district"] = user.district
    ai_result["state"] = user.state
    return jsonify(ai_result), 200


@app.route('/api/crop-advisor/history', methods=['GET'])
def crop_advisor_history():
    if 'user' not in session:
        return jsonify([])

    user = User.query.filter_by(userid=session['user']['id']).first()
    if not user:
        return jsonify([])

    # Fetch all suggestions for this user’s district & state
    records = (CropSuggestionDB.query
               .filter_by(district=user.district, state=user.state)
               .order_by(CropSuggestionDB.id.desc())
               .limit(10)
               .all())

    result = []
    for r in records:
        result.append({
            "id": r.id,
            "district": r.district,
            "state": r.state,
            "suitable_crops": r.suitable_crops,
            "reasoning": r.reasoning
        })

    return jsonify(result)


@app.route("/module/fertilizer-suggestor")
def fertilizer_ui():
    return render_template("fertilizer_advisor.html")


@app.route("/api/soil-info", methods=["GET"])
def get_soil_info():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = User.query.filter_by(userid=session["user"]["id"]).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    soil = SoilDB.query.filter_by(district=user.district, state=user.state).first()
    if not soil:
        return jsonify({"soil_type": "Unknown"})
    return jsonify({"soil_type": soil.soil_type})


# ------------------------------
# ✅ Fertilizer Advisor API
# ------------------------------
@app.route("/api/fertilizer-advisor", methods=["POST"])
def fertilizer_advisor():
    if "user" not in session:
        return jsonify({"error": "User not logged in"}), 401

    data = request.get_json()
    context = data.get("context", {})
    user_query = data.get("user_query", None)
    user_id = session["user"]["id"]

    user = User.query.filter_by(userid=user_id).first()
    if not user:
        return jsonify({"error": "User profile not found"}), 404

    # Determine soil type (auto or override)
    soil_type = context.get("soil_type")
    if not soil_type or soil_type.lower() == "auto":
        soil_record = SoilDB.query.filter_by(
            district=user.district, state=user.state
        ).first()
        soil_type = soil_record.soil_type if soil_record else "Unknown"

    # Final context for model
    full_context = {
        "district": user.district,
        "state": user.state,
        "soil_type": soil_type,
        "land_area": user.land_area,
        "crop_name": context.get("crop_name"),
        "stage": context.get("stage"),
        "budget": context.get("budget"),
        "type": context.get("type"),
        "notes": context.get("notes"),
        "user_query": user_query,
    }

    session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"

    try:
        # Call Gemini model
        result = generate_fertilizer_recommendations(full_context, session_id)
        
        
        print("Fertilizer Advisor Result:", result)

        # Store in DB
        new_record = FertilizerSuggestionDB(
            user_id=user_id,
            crop_name=context.get("crop_name"),
            soil_type=soil_type,
            conversation=result
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify(result)
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ------------------------------
# ✅ Fertilizer History API
# ------------------------------
@app.route("/api/fertilizer-advisor/history", methods=["GET"])
def fertilizer_history():
    if "user" not in session:
        return jsonify([])

    user_id = session["user"]["id"]
    records = FertilizerSuggestionDB.query.filter_by(user_id=user_id).order_by(
        FertilizerSuggestionDB.created_at.desc()
    ).limit(10).all()

    history = []
    for r in records:
        history.append({
            "id": r.id,
            "crop_name": r.crop_name,
            "soil_type": r.soil_type,
            "conversation": r.conversation
        })
    return jsonify(history)

# --------------------------
# Pest Control UI
# --------------------------
@app.route("/module/pest-control")
def pest_control_ui():
    return render_template("pest_control.html")


# --------------------------
# Pest Control Suggestion API
# --------------------------
@app.route("/api/pest-control-advisor", methods=["POST"])
def pest_control_advisor():
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    context = data.get("context", {})
    user_query = data.get("user_query", None)
    user_id = session["user"]["id"]

    user = User.query.filter_by(userid=user_id).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    soil_record = SoilDB.query.filter_by(district=user.district, state=user.state).first()
    soil_type = soil_record.soil_type if soil_record else "Unknown"

    # Final context
    full_context = {
        "district": user.district,
        "state": user.state,
        "soil_type": soil_type,
        "crop_name": context.get("crop_name"),
        "notes": context.get("notes"),
        "user_query": user_query,
    }

    session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"

    try:
        result = generate_pest_control_advice(full_context, session_id)

        new_record = PestControlDB(
            user_id=user_id,
            crop_name=context.get("crop_name"),
            pest_name=context.get("pest_name"),
            conversation=result
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify(result)

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# --------------------------
# Pest Control History
# --------------------------
@app.route("/api/pest-control-advisor/history", methods=["GET"])
def pest_control_history():
    if "user" not in session:
        return jsonify([])

    user_id = session["user"]["id"]
    records = PestControlDB.query.filter_by(user_id=user_id).order_by(
        PestControlDB.created_at.desc()
    ).limit(10).all()

    data = []
    for r in records:
        data.append({
            "crop_name": r.crop_name,
            "pest_name": r.pest_name,
            "conversation": r.conversation
        })
    return jsonify(data)


@app.route('/module/finance')
def financial_advisor_ui():
    # ensure user is logged in; if you prefer public access, remove this check
    if 'user' not in session:
        return render_template('financial_advisor.html')  # you may redirect to login in your app
    return render_template('financial_advisor.html')



@app.route('/api/financial-advisor', methods=['POST'])
def financial_advisor_api():
    # ✅ Check if user logged in
    if 'user' not in session:
        return jsonify({"error": "User not logged in"}), 401

    data = request.get_json() or {}
    context_inputs = data.get('context', {})
    user_query = data.get('user_query')
    user_id = session['user']['id']

    # ✅ Fetch user profile
    user = User.query.filter_by(userid=user_id).first()
    if not user:
        return jsonify({"error": "User profile not found"}), 404

    # ✅ Try to fetch soil details from SoilDB by district/state
    soil_info = SoilDB.query.filter_by(district=user.district, state=user.state).first()
    soil_type = soil_info.soil_type if soil_info else "Unknown"

    # ✅ Combine region and user summaries
    region = f"{user.district}, {user.state}"
    land_summary = user.land_summary or "No land summary available"
    soil_summary = user.soil_summary or "No soil summary available"

    # ✅ Construct full LLM context (enriched with user & soil details)
    full_context = {
        "user_id": user_id,
        "user_name": user.name,
        "region": region,
        "soil_type": soil_type,
        "land_area_acres": context_inputs.get('land_area') or user.land_area,
        "crop_name": context_inputs.get('crop_name'),
        "budget_preference": context_inputs.get('budget'),
        "market_preference": context_inputs.get('market_pref'),
        "additional_notes": context_inputs.get('notes'),
        "land_summary": land_summary,
        "soil_summary": soil_summary,
        "ndvi": user.ndvi,
        "ndwi": user.ndwi,
        "user_query": user_query
    }

    # ✅ Create a unique session memory ID (for multi-turn conversation)
    session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"

    try:
        # 🔹 Generate financial advice from LLM
        result = generate_financial_advice(full_context, session_id=session_id)

        # 🔹 Safely extract numeric values
        total_cost = None
        expected_profit = None
        try:
            cb = result.get("cost_breakdown", {})
            total_cost = float(cb.get("total_cost")) if isinstance(cb, dict) and "total_cost" in cb else None
            expected_profit = float(result.get("expected_profit")) if "expected_profit" in result else None
        except Exception:
            pass  # fallback gracefully

        # 🔹 Save the entire LLM conversation in DB
        record = FinancialAdvisorDB(
            user_id=str(user_id),
            crop_name=full_context.get("crop_name", "Unknown"),
            soil_type=soil_type,
            region=region,
            conversation=result,
            cost_estimate=total_cost,
            expected_profit=expected_profit,
            investment_tips="; ".join(result.get("tips", [])) if result.get("tips") else None
        )

        db.session.add(record)
        db.session.commit()

        # ✅ Return structured LLM JSON response
        return jsonify(result)

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Error generating financial advice: {str(e)}"}), 500


@app.route('/api/financial-advisor/history', methods=['GET'])
def financial_advisor_history():
    # ✅ Ensure user is logged in
    if 'user' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_id = session['user']['id']

    # ✅ Fetch user record for summaries
    user = User.query.filter_by(userid=user_id).first()
    if not user:
        return jsonify({"error": "User profile not found"}), 404

    # ✅ Fetch all financial records linked to this user
    records = (
        FinancialAdvisorDB.query
        .filter_by(user_id=str(user_id))
        .order_by(FinancialAdvisorDB.created_at.desc())
        .all()
    )

    if not records:
        return jsonify([])

    # ✅ Prepare combined structured response
    response = []
    for rec in records:
        record_data = {
            "id": rec.id,
            "crop_name": rec.crop_name,
            "soil_type": rec.soil_type,
            "region": rec.region,
            "cost_estimate": rec.cost_estimate,
            "expected_profit": rec.expected_profit,
            "investment_tips": rec.investment_tips,
            "created_at": rec.created_at.strftime("%Y-%m-%d %H:%M:%S") if rec.created_at else None,
            "conversation": rec.conversation,
            "land_summary": user.land_summary or "Not available",
            "soil_summary": user.soil_summary or "Not available"
        }
        response.append(record_data)

    return jsonify(response)


# -----------------------------------
# CREATE CROP ROOM ROUTES
# -----------------------------------

@app.route("/create-crop-room", methods=["GET"])
def create_crop_room_page():
    """Render the crop room creation form."""
    if "user" not in session:
        # Optionally redirect to login if not authenticated
        return redirect(url_for("login"))

    return render_template("create_room.html")


@app.route("/create-crop-room", methods=["GET", "POST"])
def create_crop_room():
    """Handles creation of a new crop room and initialization of AI-based steps."""

    # ✅ Ensure user logged in
    if "user" not in session:
        return redirect(url_for("login"))

    # Fetch logged-in user profile
    user = User.query.filter_by(userid=session["user"]["id"]).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 🟦 Step 1: Render creation form (GET)
    if request.method == "GET":
        return render_template("create_crop_room.html")

    # 🟨 Step 2: Handle new crop room creation (POST)
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid form data"}), 400

        # 🌍 Fetch environmental data for region
        soil_data = SoilDB.query.filter_by(district=user.district, state=user.state).first()
        weather_data = WeatherDB.query.filter_by(district=user.district, state=user.state).first()

        region = f"{user.district}, {user.state}"
        soil_type = data.get("soil_override") or (soil_data.soil_type if soil_data else "Unknown")
        weather_summary = weather_data.weather_summary if weather_data else "Not available"

        # 🧠 Additional user context
        soil_summary = user.soil_summary or "No soil summary available"
        land_summary = user.land_summary or "No land summary available"

        # 🌾 Generate AI-based initial and next steps (via Gemini 2.0)
        initial_step = generate_initial_step(
            chosen_crop=data.get("chosen_crop"),
            soil_type=soil_type,
            region=region,
            weather_summary=weather_summary,
            soil_summary=soil_summary,
            land_summary=land_summary
        )

        next_step = generate_next_step(
            chosen_crop=data.get("chosen_crop"),
            soil_type=soil_type,
            region=region,
            weather_summary=weather_summary,
            current_stage=initial_step["title"],
            soil_summary=soil_summary,
            land_summary=land_summary
        )

        # ✅ Create new CropRoom record
        new_crop_room = CropRoom(
            username=user.name,
            chosen_crop=data.get("chosen_crop"),
            region=region
        )

        # 🔹 Assign contextual and AI data
        new_crop_room.soil_description = soil_type
        new_crop_room.weather_context = weather_summary
        new_crop_room.land_area = user.land_area
        new_crop_room.suggestion = None

        # 🔹 Set initial AI stages
        new_crop_room.current_stage = initial_step["title"]
        new_crop_room.current_step = format_step_for_db(initial_step)
        new_crop_room.next_step = format_step_for_db(next_step)
        new_crop_room.previous_steps = []

        # 🔹 Timeline tracking
        new_crop_room.timeline = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "Crop Room Created",
            "details": f"AI initialized with step: {initial_step['title']} → Next: {next_step['title']}."
        }]

        # 🔹 Optional data
        new_crop_room.user_notes = [{
    "created": datetime.now(timezone.utc).isoformat(),
    "note": data.get("notes", "")
}]
        new_crop_room.ai_doubt_history = []
        new_crop_room.budget_breakdown = {
            "estimated_budget": data.get("budget") or "Not provided",
            "expected_yield": data.get("expectation") or "N/A"
        }

        # 💾 Commit to DB
        db.session.add(new_crop_room)
        db.session.commit()

        # ✅ Return success + redirect URL
        return jsonify({
            "status": "success",
            "crop_id": new_crop_room.crop_id,
            "redirect_url": f"/crop-room-result/{new_crop_room.crop_id}"
        })




@app.route("/crop-room-result/<crop_id>")
def crop_room_result(crop_id):
    """Display full analysis and AI context for a specific crop room."""
    print(f"[DEBUG] Fetching Crop Room with ID: {crop_id}")
    room = CropRoom.query.filter_by(crop_id=crop_id).first()
    if not room:
        return render_template("404.html"), 404
    return render_template("croproom.html", room=room)


@app.route("/update-step/<crop_id>", methods=["POST"])
def update_step(crop_id):
    """Completes current step, promotes next, and generates a new next step via Gemini."""
    
    room = CropRoom.query.filter_by(crop_id=crop_id).first()
    if not room:
        return jsonify({"status": "error", "error": "Crop room not found"}), 404

    user = User.query.filter_by(name=room.username).first()
    if not user:
        return jsonify({"status": "error", "error": "User not found"}), 404

    # Ensure required fields exist
    if not room.current_step or not room.next_step:
        return jsonify({"status": "error", "error": "AI steps not initialized"}), 400

    # 🧩 Step 1: Move current_step → previous_steps
    if not room.previous_steps:
        room.previous_steps = []
    
    room.previous_steps.append({
        "step": room.current_step,
        "completed_at": datetime.now(timezone.utc).isoformat()
    })

    # 🧩 Step 2: Promote next_step → current_step
    room.current_step = room.next_step
    room.current_stage = room.next_step.get("title", "Ongoing Stage")

    # 🧩 Step 3: Generate new next_step from Gemini
    try:
        new_next = generate_next_step(
            chosen_crop=room.chosen_crop,
            soil_type=room.soil_description,
            region=room.region,
            weather_summary=room.weather_context,
            current_stage=room.current_step.get("title"),
            soil_summary=user.soil_summary or "N/A",
            land_summary=user.land_summary or "N/A"
        )
        room.next_step = format_step_for_db(new_next)
    except Exception as e:
        print(f"[Gemini Error] Failed to generate next step: {e}")
        room.next_step = format_step_for_db({
            "title": "Next Stage Pending",
            "description": "AI could not generate the next step. Try again later."
        })

    # 🧩 Step 4: Add a new timeline event
    if not room.timeline:
        room.timeline = []
    room.timeline.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "Step Completed",
        "details": f"Moved to '{room.current_stage}'. Next: '{room.next_step['title']}'."
    })

    # 🧩 Step 5: Save changes
    db.session.commit()

    # 🧩 Step 6: Return updated state to frontend
    return jsonify({
        "status": "success",
        "current_step": room.current_step,
        "next_step": room.next_step,
        "previous_steps": room.previous_steps,
        "timeline": room.timeline
    })
    

# Fertilizer AI
@app.route("/api/fertilizer/<crop_id>", methods=["POST"])
def api_fertilizer(crop_id):
    """Generate context-aware fertilizer recommendations using Gemini AI."""
    room = CropRoom.query.filter_by(crop_id=crop_id).first()
    if not room:
        return jsonify({"error": "Crop room not found"}), 404

    user = User.query.filter_by(name=room.username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 🌍 Fetch soil and weather context
    soil_data = SoilDB.query.filter_by(district=user.district, state=user.state).first()
    weather_data = WeatherDB.query.filter_by(district=user.district, state=user.state).first()

    context = {
        "region": f"{user.district}, {user.state}",
        "soil_type": soil_data.soil_type if soil_data else room.soil_description,
        "ph_level": getattr(soil_data, "ph_level", "Unknown"),
        "soil_summary": user.soil_summary or "No soil summary available.",
        "land_summary": user.land_summary or "No land summary available.",
        "avg_temp": getattr(weather_data, "avg_temp", "N/A"),
        "humidity": getattr(weather_data, "humidity", "N/A"),
        "rainfall": getattr(weather_data, "rainfall", "N/A"),
        "weather_summary": getattr(weather_data, "weather_summary", room.weather_context),
        "land_area": user.land_area or "Unknown",
        "ndvi": user.ndvi or "N/A",
        "ndwi": user.ndwi or "N/A"
    }

    # 🧠 Call Gemini AI for fertilizer recommendations
    try:
        ai_result = get_fertilizer_suggestions(
            crop=room.chosen_crop,
            context=context
        )
    except Exception as e:
        print(f"[Gemini Fertilizer Error]: {e}")
        return jsonify({"error": "AI generation failed. Please try again."}), 500

    # ✅ Update DB with AI-generated recommendations
    room.fertilizers_suggested = ai_result

    # 🧩 Ensure timeline is initialized
    if not room.timeline:
        room.timeline = []

    # 🕒 Append new event (aligned with unified event style)
    room.timeline.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "Fertilizer Advice Generated",
        "details": (
            f"AI provided {len(ai_result)} fertilizer suggestions for "
            f"'{room.chosen_crop}' based on soil type '{context['soil_type']}', "
            f"rainfall {context['rainfall']} mm, and region '{context['region']}'."
        )
    })

    # 💾 Commit all updates
    db.session.commit()


    # ✅ Send back data to frontend
    return jsonify({
        "context_used": context,
        "recommendations": ai_result
    })



# 🐛 PEST AI MODULE
@app.route("/api/pest/<crop_id>", methods=["POST"])
def api_pest(crop_id):
    """Run AI pest risk detection and management advice."""
    room = CropRoom.query.filter_by(crop_id=crop_id).first()
    if not room:
        return jsonify({"error": "Crop room not found"}), 404

    user = User.query.filter_by(name=room.username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 🌦️ Get soil and weather data
    weather_data = WeatherDB.query.filter_by(district=user.district, state=user.state).first()
    soil_data = SoilDB.query.filter_by(district=user.district, state=user.state).first()

    context = {
        "crop": room.chosen_crop,
        "region": f"{user.district}, {user.state}",
        "soil_type": soil_data.soil_type if soil_data else "Unknown",
        "avg_temp": getattr(weather_data, "avg_temp", "N/A"),
        "humidity": getattr(weather_data, "humidity", "N/A"),
        "rainfall": getattr(weather_data, "rainfall", "N/A"),
        "weather_summary": getattr(weather_data, "weather_summary", room.weather_context),
        "land_summary": user.land_summary or "N/A",
        "soil_summary": user.soil_summary or "N/A",
        "current_stage": room.current_stage or "Unknown"
    }

    # 🤖 Get AI-generated pest guidelines
    try:
        ai_result = get_pest_guidelines(crop=room.chosen_crop, context=context)
    except Exception as e:
        print(f"[Gemini Pest Error]: {e}")
        return jsonify({"error": "AI generation failed. Please try again."}), 500

    # 🧩 Save pest analysis and update timeline
    room.pest_guideline = ai_result
    if not isinstance(room.timeline, list):
        room.timeline = []

    room.timeline.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "AI Pest Analysis Completed",
        "details": f"AI detected pest risks for '{room.chosen_crop}' at '{room.current_stage}' stage in {context['region']}."
    })

    db.session.commit()

    return jsonify({
        "context_used": context,
        "pest_analysis": ai_result
    })

# 💰 FINANCIAL AI MODULE
@app.route("/api/finance/<crop_id>", methods=["POST"])
def api_finance(crop_id):
    """AI-driven farm budgeting and financial advice."""
    room = CropRoom.query.filter_by(crop_id=crop_id).first()
    if not room:
        return jsonify({"error": "Crop room not found"}), 404

    user = User.query.filter_by(name=room.username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 🌾 Build analysis context
    context = {
        "region": room.region or f"{user.district}, {user.state}",
        "soil_type": room.soil_description or "Unknown",
        "weather_summary": room.weather_context or "N/A",
        "expected_yield": (room.budget_breakdown or {}).get("expected_yield", "N/A"),
        "estimated_budget": (room.budget_breakdown or {}).get("estimated_budget", "Unknown"),
        "land_area": user.land_area or "Unknown",
        "ndvi": user.ndvi or "N/A",
        "ndwi": user.ndwi or "N/A"
    }

    # 🤖 Generate financial suggestions
    try:
        ai_result = get_financial_advice(crop=room.chosen_crop, context=context)
    except Exception as e:
        print(f"[Gemini Finance Error]: {e}")
        return jsonify({"error": "AI generation failed. Please try again."}), 500

    # ✅ Update DB
    room.financial_suggestion = ai_result
    if not isinstance(room.timeline, list):
        room.timeline = []

    room.timeline.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "AI Financial Advice Generated",
        "details": (
            f"Financial analysis done for {room.chosen_crop}. "
            f"Budget ₹{context['estimated_budget']} | Expected yield: {context['expected_yield']} quintals."
        )
    })

    db.session.commit()

    return jsonify({
        "context_used": context,
        "financial_advice": ai_result
    })



# 🤖 AI Doubt Solver
@app.route("/api/ai-doubt/<crop_id>", methods=["POST"])
def api_ai_doubt(crop_id):
    """Chat-like AI doubt resolution module."""
    room = CropRoom.query.filter_by(crop_id=crop_id).first()
    if not room:
        return jsonify({"error": "Crop room not found"}), 404

    user = User.query.filter_by(name=room.username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    context = {
        "crop": room.chosen_crop,
        "soil_type": room.soil_description,
        "weather_summary": room.weather_context,
        "region": room.region,
        "current_stage": room.current_stage,
        "land_area": user.land_area or "Unknown",
        "soil_summary": user.soil_summary or "N/A",
        "land_summary": user.land_summary or "N/A"
    }

    ai_reply = get_ai_doubt_response(query=query, context=context)

    if not room.ai_doubt_history:
        room.ai_doubt_history = []
    room.ai_doubt_history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": query,
        "ai_reply": ai_reply
    })

    room.timeline.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "AI Doubt Answered",
        "details": f"AI responded to query: '{query[:40]}...'"
    })
    db.session.add(room)
    db.session.commit()
    return jsonify({
        "reply": ai_reply,
        "context_used": context
    })


# ----------------------------------------------------------
# 🌿 Route: Web UI
# ----------------------------------------------------------
@app.route("/module/disease-detection")
def disease_page():
    return render_template("disease_detection.html")

# ----------------------------------------------------------
# 🧬 Route: API Endpoint for Disease Detection
# ----------------------------------------------------------
@app.route("/api/disease-detect", methods=["POST"])
def detect_disease():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        crop_type = request.form.get("crop_type", "Unknown Crop")
        preference = request.form.get("preference", "Balanced")

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            yolo_result = run_yolo_classification(tmp.name)

        if "error" in yolo_result:
            return jsonify(yolo_result), 500

        top1 = yolo_result["top1"]
        top5 = yolo_result["top5"]

        # Get Gemini guidance
        guidance = get_gemini_guidance(top5, crop_type, preference)

        response = {
            "task": "disease_classification_with_guidance",
            "crop_type": crop_type,
            "preference": preference,
            "disease_detected": top1["class"],
            "confidence": top1["confidence"],
            "top5_predictions": top5,
            "ai_guidance": guidance
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

    
@app.route("/logout", methods=["GET", "POST"])
def logout():
    """Logout user, clear session, and redirect to login."""
    # Option 1: If you’re using a web UI (redirect to login)
    if request.method == "GET":
        user_email = session.get("user", {}).get("email") if "user" in session else None

        # Log optional user info before logout
        app.logger.info(f"User logged out: {user_email or 'Unknown'}")

        # Clear all session data
        session.clear()

        # Redirect to login page (or your home)
        return redirect(url_for("home"))  # assumes you have a `login` route

    # Option 2: If frontend calls via AJAX / fetch
    elif request.method == "POST":
        session.clear()
        return jsonify({"message": "Logout successful"}), 200
    
# 🧠 AI EXPLAINER
@app.route("/module/ai-explainer")
def ai_explainer():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("ai_explainer.html")


# ==========================================
# 🧠 AI EXPLAINER API
# ==========================================
@app.route("/api/ai-explainer", methods=["POST"])
def ai_explainer_api():
    """
    AI Explainer Endpoint
    - Takes user query + language
    - Uses user DB context (optional enhancement)
    - Returns structured lesson + audio
    """

    # 🔐 Optional: Require login (recommended for your app)
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}

        user_input = data.get("user_input")
        language = data.get("language", "english")

        if not user_input:
            return jsonify({"error": "user_input is required"}), 400

        # ==========================================
        # 🧠 FETCH USER CONTEXT (SMART INTEGRATION)
        # ==========================================
        user = User.query.filter_by(userid=session["user"]["id"]).first()

        user_context = {}

        if user:
            user_context = {
                "location": f"{user.district}, {user.state}",
                "land_area": user.land_area,
                "soil_summary": user.soil_summary,
                "land_summary": user.land_summary,
                "ndvi": user.ndvi,
                "ndwi": user.ndwi
            }

            # 🔎 Add soil DB context
            soil = SoilDB.query.filter_by(
                district=user.district,
                state=user.state
            ).first()

            if soil:
                user_context.update({
                    "soil_type": soil.soil_type,
                    "soil_ph": soil.ph_level,
                    "nitrogen": soil.nitrogen,
                    "phosphorus": soil.phosphorus,
                    "potassium": soil.potassium
                })

            # 🌦️ Add weather context
            weather = WeatherDB.query.filter_by(
                district=user.district,
                state=user.state
            ).first()

            if weather:
                user_context.update({
                    "temperature": weather.avg_temp,
                    "rainfall": weather.rainfall,
                    "humidity": weather.humidity,
                    "weather_summary": weather.weather_summary
                })

        # ==========================================
        # 🚀 RUN AI ENGINE
        # ==========================================
        result = run_ai_explainer(
            user_input=user_input,
            language=language,
            user_data=user_context
        )

        return jsonify(result), 200

    except Exception as e:
        print("[AI EXPLAINER ERROR]:", str(e))
        return jsonify({
            "error": "AI processing failed",
            "details": str(e)
        }), 500


@app.route("/module/equipment")
def equipment():
    return render_template("equipment_advisor.html")


# =========================================================
# 🚜 EQUIPMENT RECOMMENDATION API
# =========================================================
@app.route("/api/equipment-recommend", methods=["POST"])
def equipment_recommend():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}

        # 🔹 CONTEXT (from frontend)
        context = {
            "crop": data.get("crop"),
            "land_size": data.get("land_size"),
            "soil_type": data.get("soil_type"),
            "location": data.get("location"),
            "budget": data.get("budget"),
            "farming_type": data.get("farming_type"),
            "irrigation": data.get("irrigation")
        }

        user_query = data.get("query")

        # 🚀 CALL YOUR GEMINI FUNCTION
        result = generate_equipment_recommendations(
            context=context,
            user_query=user_query
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Failed to generate recommendations",
            "details": str(e)
        }), 500

# ⚖️ DECISION COMPARISON
@app.route("/module/decision")
def decision():
    return render_template("decision_comparison.html")

# ==========================================
# ⚖️ DECISION COMPARISON API
# ==========================================
@app.route("/api/decision", methods=["POST"])
def decision_api():
    """
    Decision Comparison API (Simplified)
    - Input: crops + language
    - Context: auto from DB (user + soil + weather)
    """

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}

        # ✅ ONLY INPUTS
        crops = data.get("crops", ["Rice", "Wheat", "Maize"])
        language = data.get("context", {}).get("language", "english")

        # ==========================================
        # 👤 FETCH USER
        # ==========================================
        user = User.query.filter_by(userid=session["user"]["id"]).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # ==========================================
        # 🌍 BASE CONTEXT (AUTO)
        # ==========================================
        context = {
            "land_area": f"{user.land_area} acres" if user.land_area else "1 acres",
            "soil_type": "loam",
            "soil_ph": 6.5,
            "rainfall": "medium",
            "temperature": "moderate",
            "location": f"{user.district}, {user.state}",
            "language": language
        }

        # ==========================================
        # 🌱 SOIL ENRICHMENT
        # ==========================================
        soil = SoilDB.query.filter_by(
            district=user.district,
            state=user.state
        ).first()

        if soil:
            context["soil_type"] = soil.soil_type or context["soil_type"]
            context["soil_ph"] = soil.ph_level or context["soil_ph"]

        # ==========================================
        # 🌦️ WEATHER ENRICHMENT
        # ==========================================
        weather = WeatherDB.query.filter_by(
            district=user.district,
            state=user.state
        ).first()

        if weather:
            if weather.rainfall:
                context["rainfall"] = (
                    "high" if weather.rainfall > 1000 else
                    "low" if weather.rainfall < 500 else
                    "medium"
                )

            if weather.avg_temp:
                context["temperature"] = (
                    "hot" if weather.avg_temp > 30 else
                    "cool" if weather.avg_temp < 20 else
                    "moderate"
                )

        # ==========================================
        # 🚀 RUN ENGINE
        # ==========================================
        result = run_decision_engine(
            crops=crops,
            user_data=context
        )

        # ==========================================
        # ✅ RESPONSE FORMAT (MATCH FRONTEND)
        # ==========================================
        return jsonify({
            "language": language,
            "output": result,          # 🔥 IMPORTANT FIX
            "context_used": context    # optional (debug / UI)
        }), 200

    except Exception as e:
        print("[DECISION API ERROR]:", str(e))
        return jsonify({
            "error": "Decision processing failed",
            "details": str(e)
        }), 500

# 📦 INVENTORY
@app.route("/module/inventory")
def inventory():
    return render_template("inventory.html")

@app.route("/api/inventory", methods=["GET"])
def get_inventory():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user"]["id"]

    record = InventoryStore.query.filter_by(user_id=user_id).first()

    if not record:
        return jsonify({"inventory": []})

    return jsonify({
        "inventory": record.data.get("items", [])
    })


# ============================================================
# ➕ ADD ITEM
# ============================================================
@app.route("/api/inventory/add", methods=["POST"])
def add_inventory():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user"]["id"]
    data = request.get_json()

    record = InventoryStore.query.filter_by(user_id=user_id).first()

    if not record:
        record = InventoryStore(
            user_id=user_id,
            data={"items": []}
        )
        db.session.add(record)

    item = {
        "id": int(time.time() * 1000),
        "name": data.get("name"),
        "category": data.get("category"),
        "unit": data.get("unit"),
        "quantity": data.get("quantity", 0),
        "threshold": data.get("threshold", 0),
        "usage_rate": data.get("usage_rate", 0),
        "expiry": data.get("expiry"),
        "created_at": datetime.utcnow().isoformat()
    }

    record.data["items"].append(item)
    db.session.commit()

    return jsonify({"message": "Item added"})


# ============================================================
# 🔽 USE ITEM
# ============================================================
@app.route("/api/inventory/use", methods=["POST"])
def use_inventory():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user"]["id"]
    data = request.get_json()

    record = InventoryStore.query.filter_by(user_id=user_id).first()

    if not record:
        return jsonify({"error": "Inventory not found"}), 404

    for item in record.data["items"]:
        if item["id"] == data.get("id"):
            item["quantity"] = max(0, item["quantity"] - 1)

    db.session.commit()

    return jsonify({"message": "Updated"})


# ============================================================
# ❌ DELETE ITEM
# ============================================================
@app.route("/api/inventory/delete", methods=["POST"])
def delete_inventory():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user"]["id"]
    data = request.get_json()

    record = InventoryStore.query.filter_by(user_id=user_id).first()

    if not record:
        return jsonify({"error": "Inventory not found"}), 404

    record.data["items"] = [
        i for i in record.data["items"]
        if i["id"] != data.get("id")
    ]

    db.session.commit()

    return jsonify({"message": "Deleted"})


# ============================================================
# ⚠️ RISK + ALERT ENGINE
# ============================================================
@app.route("/api/inventory/alerts", methods=["GET"])
def inventory_alerts():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user"]["id"]

    record = InventoryStore.query.filter_by(user_id=user_id).first()

    if not record:
        return jsonify({"alerts": []})

    alerts = []

    for item in record.data.get("items", []):

        # LOW STOCK
        if item["quantity"] <= item["threshold"]:
            alerts.append({
                "type": "low",
                "message": f"{item['name']} is low"
            })

        # RUN OUT SOON
        if item.get("usage_rate"):
            days = item["quantity"] / (item["usage_rate"] or 1)
            if days < 3:
                alerts.append({
                    "type": "risk",
                    "message": f"{item['name']} will run out in {round(days,1)} days"
                })

        # EXPIRY
        if item.get("expiry"):
            try:
                days_left = (datetime.fromisoformat(item["expiry"]) - datetime.utcnow()).days
                if days_left < 5:
                    alerts.append({
                        "type": "expiry",
                        "message": f"{item['name']} expiring soon"
                    })
            except:
                pass

    return jsonify({"alerts": alerts})


# ============================================================
# 🧠 AI SUMMARY (LLM READY)
# ============================================================
@app.route("/api/inventory/summary", methods=["POST"])
def inventory_summary():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        inventory = data.get("inventory", [])

        # 🚀 CALL GENAI
        result = generate_inventory_summary(inventory)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Failed to generate summary",
            "details": str(e)
        }), 500


# 🏪 MARKETPLACE
@app.route("/module/marketplace")
def marketplace():
    return render_template("community_marketplace.html")

# ==========================================
# 📍 ADD NEARBY SERVICE
# ==========================================
@app.route("/api/nearby/add", methods=["POST"])
def add_nearby_service():

    data = request.get_json()

    try:
        # 🔍 VALIDATION
        required_fields = ["name", "type", "lat", "lng"]
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"{field} is required"}), 400
            
        user = User.query.filter_by(userid=session["user"]["id"]).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        

        # 🏷️ CLEAN TAGS
        tags = data.get("tags", [])
        if isinstance(tags, list):
            tags = [t.strip().lower() for t in tags if t.strip()]
        else:
            tags = []

        # 📦 CREATE OBJECT
        service = NearbyService(
            user_id=user.userid,  # 🔥 later replace with session user
            name=data["name"],
            service_type=data["type"],
            description=data.get("description"),
            latitude=float(data["lat"]),
            longitude=float(data["lng"]),
            contact_number=data.get("contact"),
            address=data.get("address"),
            tags=tags,
            source="user"
        )

        db.session.add(service)
        db.session.commit()

        return jsonify({
            "message": "Service added successfully",
            "service_id": service.id
        })

    except Exception as e:
        print("[ADD SERVICE ERROR]:", str(e))
        return jsonify({"error": str(e)}), 500


# 📍 NEARBY SERVICES
@app.route("/nearby-services")
def nearby_services():
    return render_template("nearby_service.html")

# ==========================================
# 📍 NEARBY SERVICES API (DB + EXTERNAL)
# ==========================================
from math import radians, cos, sin, asin, sqrt

def calc_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    return R * c


@app.route("/api/nearby-services", methods=["GET"])
def nearby_services_api():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user = User.query.filter_by(userid=session["user"]["id"]).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        lat, lon = user.latitude, user.longitude

        if lat is None or lon is None:
            return jsonify({"error": "User location not available"}), 400

        # ==========================================
        # 🔹 1. FETCH FROM DB
        # ==========================================
        db_services = NearbyService.query.all()
        db_results = []

        for s in db_services:
            try:
                # ✅ FIX TAGS (handles JSON + string)
                tags = s.tags or []
                if isinstance(tags, str):
                    import json
                    try:
                        tags = json.loads(tags)
                    except:
                        tags = []

                distance = calc_distance(lat, lon, s.latitude, s.longitude)

                db_results.append({
                    "name": s.name,
                    "lat": s.latitude,
                    "lng": s.longitude,
                    "distance_km": round(distance, 2),
                    "score": round(1 / (1 + distance), 3),
                    "source": "cropai_store_db",
                    "tags": [t.lower() for t in tags],
                    "type": (s.service_type or "").lower()
                })

            except Exception as e:
                print("[DB ITEM ERROR]:", e)

        # ==========================================
        # 🔹 2. FETCH EXTERNAL (SAFE)
        # ==========================================
        try:
            external_services = find_nearby_services(lat, lon)
            if not isinstance(external_services, list):
                external_services = []
        except Exception as e:
            print("[EXTERNAL ERROR]:", e)
            external_services = []

        # ==========================================
        # 🔹 3. MERGE
        # ==========================================
        all_services = db_results + external_services
        print(f"[DEBUG] Total services: {len(all_services)}")

        # ==========================================
        # 🔹 4. FIXED GROUPING
        # ==========================================
        def group_services(services, keyword):
            result = []

            for s in services:
                try:
                    type_text = s.get("type", "")
                    tags = s.get("tags", [])

                    if isinstance(tags, str):
                        import json
                        try:
                            tags = json.loads(tags)
                        except:
                            tags = []

                    tags_text = " ".join(tags)

                    full_text = (type_text + " " + tags_text).lower()

                    if keyword in full_text:
                        result.append(s)

                except Exception as e:
                    print("[GROUP ERROR]:", e)

            return result

        fertilizer_shops = sorted(
            group_services(all_services, "fertilizer"),
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:10]

        markets = sorted(
            group_services(all_services, "market"),
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:10]

        warehouses = sorted(
            group_services(all_services, "warehouse"),
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:10]
        
        print(f"[DEBUG] Fertilizer Shops: {len(fertilizer_shops)}, Markets: {len(markets)}, Warehouses: {len(warehouses)}")

        # ==========================================
        # 🔹 5. RESPONSE
        # ==========================================
        return jsonify({
            "location": {
                "lat": lat,
                "lon": lon
            },
            "fertilizer_shops": fertilizer_shops,
            "markets": markets,
            "warehouses": warehouses
        })

    except Exception as e:
        print("[NEARBY SERVICES ERROR]:", str(e))
        return jsonify({
            "error": "Failed to fetch services",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # creates all tables
    app.run(debug=True)
