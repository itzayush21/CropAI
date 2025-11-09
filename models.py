from flask_sqlalchemy import SQLAlchemy
import uuid
from datetime import datetime

db = SQLAlchemy()

# ---------------------------
# USER TABLE
# ---------------------------
class User(db.Model):
    __tablename__ = 'users'
    userid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    land_area = db.Column(db.Float, nullable=True)
    ndvi = db.Column(db.Float, nullable=True)
    ndwi = db.Column(db.Float, nullable=True)
    land_summary = db.Column(db.Text, nullable=True)
    soil_summary = db.Column(db.Text, nullable=True)
    district = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)

# ---------------------------
# SOIL DATABASE
# ---------------------------
class SoilDB(db.Model):
    __tablename__ = 'soil_db'
    id = db.Column(db.Integer, primary_key=True)
    district = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    soil_type = db.Column(db.String(100))
    ph_level = db.Column(db.Float)
    organic_carbon = db.Column(db.Float)
    nitrogen = db.Column(db.Float)
    phosphorus = db.Column(db.Float)
    potassium = db.Column(db.Float)

# ---------------------------
# WEATHER DATABASE
# ---------------------------
class WeatherDB(db.Model):
    __tablename__ = 'weather_db'
    id = db.Column(db.Integer, primary_key=True)
    district = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    avg_temp = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    humidity = db.Column(db.Float)
    weather_summary = db.Column(db.Text)

# ---------------------------
# CROP SUGGESTION DATABASE
# ---------------------------
class CropSuggestionDB(db.Model):
    __tablename__ = 'crop_suggestion_db'
    id = db.Column(db.Integer, primary_key=True)
    district = db.Column(db.String(100))
    state = db.Column(db.String(100))
    suitable_crops = db.Column(db.JSON)  # changed from Text → JSON
    reasoning = db.Column(db.Text)


# ---------------------------
# FERTILIZER SUGGESTION DATABASE
# ---------------------------
class FertilizerSuggestionDB(db.Model):
    __tablename__ = 'fertilizer_suggestion_db'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), nullable=False)
    crop_name = db.Column(db.String(100))
    soil_type = db.Column(db.String(100))
    conversation = db.Column(db.JSON)  # 🔥 store entire chat or Gemini response
    created_at = db.Column(db.DateTime, default=db.func.now())



# ---------------------------
# PEST CONTROL DATABASE
# ---------------------------
class PestControlDB(db.Model):
    __tablename__ = 'pest_control_db'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), nullable=False)   # ✅ links conversation to the logged-in user
    crop_name = db.Column(db.String(100))
    pest_name = db.Column(db.String(150))                # ✅ optionally store pest queried
    conversation = db.Column(db.JSON)                    # ✅ full LLM response/conversation context
    created_at = db.Column(db.DateTime, default=db.func.now())


# ---------------------------
# FINANCIAL ADVISOR DATABASE
# ---------------------------
class FinancialAdvisorDB(db.Model):
    __tablename__ = 'financial_advisor_db'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), nullable=False)  # ✅ Links to logged-in user
    crop_name = db.Column(db.String(100), nullable=False)
    soil_type = db.Column(db.String(100))
    region = db.Column(db.String(100))
    
    # Structured LLM output (e.g., budget breakdown, ROI analysis, etc.)
    conversation = db.Column(db.JSON)  # ✅ Store entire conversation with model
    cost_estimate = db.Column(db.Float)
    expected_profit = db.Column(db.Float)
    investment_tips = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.now())


# ---------------------------
# CROP ROOM DATABASE
# ---------------------------
class CropRoom(db.Model):
    __tablename__ = 'crop_room_db'

    id = db.Column(db.Integer, primary_key=True)
    crop_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    username = db.Column(db.String(100), nullable=False)
    chosen_crop = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # -------------------------------
    # CONTEXTUAL DETAILS
    # -------------------------------
    region = db.Column(db.String(150))
    soil_description = db.Column(db.Text)
    weather_context = db.Column(db.Text)
    land_area = db.Column(db.Float)

    # -------------------------------
    # AI MODULE OUTPUTS
    # -------------------------------
    suggestion = db.Column(db.JSON)             # Crop suggestion / summary
    fertilizers_suggested = db.Column(db.JSON)  # Fertilizer insights
    pest_guideline = db.Column(db.JSON)         # Pest control data
    financial_suggestion = db.Column(db.JSON)   # Financial AI output
    budget_breakdown = db.Column(db.JSON)       # Cost & yield info

    # -------------------------------
    # TIMELINE & PIPELINE STEPS
    # -------------------------------
    current_step = db.Column(db.JSON)           # ✅ Active step (AI generated)
    next_step = db.Column(db.JSON)              # ✅ Upcoming AI-predicted step
    previous_steps = db.Column(db.JSON, default=[])  # ✅ Completed steps
    current_stage = db.Column(db.String(200))   # e.g. "Germination", "Harvest"
    timeline = db.Column(db.JSON, default=list)   # ✅ Event/activity timeline

    # -------------------------------
    # NOTES & INTERACTIONS
    # -------------------------------
    user_notes = db.Column(db.JSON, default=[])         # ✅ Farmer notes
    ai_doubt_history = db.Column(db.JSON, default=[])   # ✅ AI chat history

    # -------------------------------
    # INITIALIZER
    # -------------------------------
    def __init__(self, username, chosen_crop, region=None):
        self.username = username
        self.chosen_crop = chosen_crop
        self.region = region
        self.crop_id = f"CRP-{uuid.uuid4().hex[:10].upper()}"
        self.timeline = []
        self.previous_steps = []
        self.user_notes = []
        self.ai_doubt_history = []