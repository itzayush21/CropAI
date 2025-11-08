from flask import Flask,render_template,request,redirect, session, flash, url_for, jsonify
from config import Config
from models import db
from auth.auth_client import create_supabase_client
from module.preprocessing import enrich_user_data,get_lat_lon_from_location
from models import db, User, SoilDB, WeatherDB, CropSuggestionDB, FertilizerSuggestionDB, PestControlDB, FinancialAdvisorDB, CropRoom
from module.pest_control_genai import generate_pest_control_advice
from module.genai_crop_advisor import generate_crop_suggestions
from module.fertilizer_genai import generate_fertilizer_recommendations
from module.financial_genai import generate_financial_advice
import uuid
import math
import json
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
        except Exception as e:
            flash(f"Signup failed: {e}", "error")
            return render_template("register.html")

        if not res.user:
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
    """
    Handles both:
    - GET: returns context info (soil/weather)
    - POST: creates a new crop room with base context & redirects to /crop-room-result/<crop_id>
    """
    # ✅ Ensure user logged in
    if "user" not in session:
        return redirect(url_for("login"))

    # Fetch user profile
    user = User.query.filter_by(userid=session["user"]["id"]).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # 🟩 Step 1: Return context info (GET)
    if request.method == "GET" and request.args.get("info") == "true":
        soil_data = SoilDB.query.filter_by(district=user.district, state=user.state).first()
        weather_data = WeatherDB.query.filter_by(district=user.district, state=user.state).first()

        context = {
            "region": f"{user.district}, {user.state}",
            "soil_type": soil_data.soil_type if soil_data else "Unknown",
            "ph_level": soil_data.ph_level if soil_data else None,
            "weather_summary": weather_data.weather_summary if weather_data else "Not available",
            "avg_temp": weather_data.avg_temp if weather_data else None,
            "rainfall": weather_data.rainfall if weather_data else None,
            "humidity": weather_data.humidity if weather_data else None
        }
        return jsonify(context)

    # 🟦 Step 2: Render the creation form (GET)
    if request.method == "GET":
        return render_template("create_crop_room.html")

    # 🟨 Step 3: Handle form submission (POST)
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid form data"}), 400

        # Fetch soil & weather data for the user’s region
        soil_data = SoilDB.query.filter_by(district=user.district, state=user.state).first()
        weather_data = WeatherDB.query.filter_by(district=user.district, state=user.state).first()

        # 🌍 Contextual details
        region = f"{user.district}, {user.state}"
        soil_type = data.get("soil_override") or (soil_data.soil_type if soil_data else "Unknown")
        weather_context = weather_data.weather_summary if weather_data else "Not available"

        # 🌾 Placeholder AI-based suggestion
        suggestion = {
            "summary": f"{data.get('chosen_crop')} is suitable for {region} based on soil ({soil_type}) and weather.",
            "key_recommendations": [
                "Ensure proper irrigation at vegetative stage.",
                "Monitor pest activity bi-weekly.",
                "Add organic compost before sowing."
            ]
        }

        # 🕒 Initialize timeline
        timeline_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "Crop Room Created",
            "details": f"Initial analysis generated for {data.get('chosen_crop')}."
        }

        # Create the new CropRoom record
        new_crop_room = CropRoom(
            username=user.name,
            chosen_crop=data.get("chosen_crop"),
            region=region
        )

        new_crop_room.soil_description = soil_type
        new_crop_room.weather_context = weather_context
        new_crop_room.land_area = user.land_area or None
        new_crop_room.suggestion = suggestion
        new_crop_room.current_stage = "Initial Setup"
        new_crop_room.timeline = [timeline_entry]
        new_crop_room.previous_steps = []
        new_crop_room.user_notes = [{"created": datetime.utcnow().isoformat(), "note": data.get("notes", "")}]
        new_crop_room.ai_doubt_history = []
        new_crop_room.budget_breakdown = {
            "estimated_budget": data.get("budget") or "Not provided",
            "expected_yield": data.get("expectation") or "N/A"
        }

        # Save to DB
        db.session.add(new_crop_room)
        db.session.commit()

        # ✅ Send response with redirect URL
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



if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # creates all tables
    app.run(debug=True)
