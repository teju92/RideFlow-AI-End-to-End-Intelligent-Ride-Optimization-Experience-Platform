
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import json
from groq import Groq

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title       = "RideFlow AI API",
    description = "End-to-End Intelligent Ride Optimization Platform",
    version     = "1.0.0"
)

# ── Load models ───────────────────────────────────────────────
demand_model   = joblib.load('demand_prediction_model.pkl')
supply_model   = joblib.load('driver_supply_model.pkl')
cancel_model   = joblib.load('cancellation_model.pkl')
driver_profile = joblib.load('driver_profile.pkl')

with open('feedback_results.json', 'r') as f:
    feedback_map = json.load(f)

client = Groq(api_key="your_groq_api_key_here")

# ── Request Models ────────────────────────────────────────────
class DemandRequest(BaseModel):
    pickup_cluster : int
    hour           : int
    lag_1h         : float = 45
    lag_2h         : float = 40

class SupplyRequest(BaseModel):
    hour              : int
    day_of_week       : int
    is_weekend        : int
    cluster_avg_supply: float = 25
    supply_lag_1h     : float = 20
    supply_lag_2h     : float = 18
    traffic_level_enc : int   = 1

class CancelRequest(BaseModel):
    hour              : int
    day_of_week       : int
    is_weekend        : int
    traffic_level_enc : int
    gap               : float
    surge_multiplier  : float
    pickup_cluster    : int

class FeedbackRequest(BaseModel):
    feedback_text: str

class MatchRequest(BaseModel):
    customer_location : str   = "Chennai Central"
    hour              : int   = 18
    traffic           : str   = "medium"
    min_rating        : float = 3.5
    max_eta           : float = 20.0
    max_cancel_rate   : float = 0.4

class ChatRequest(BaseModel):
    message  : str
    language : str = "English"
    role     : str = "Customer"

# ── Helper ────────────────────────────────────────────────────
def calc_surge(gap):
    if gap <= 0:    return 1.0
    elif gap <= 5:  return round(1.0 + (gap/5)*0.5, 2)
    elif gap <= 15: return round(1.5 + (gap-5)/10,  2)
    else:           return 3.0

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── Root ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "platform" : "RideFlow AI",
        "version"  : "1.0.0",
        "endpoints": [
            "/predict/demand",
            "/predict/supply",
            "/predict/cancel",
            "/predict/gap",
            "/analyze/feedback",
            "/match/driver",
            "/chat"
        ]
    }

# ── 1. Demand Prediction ──────────────────────────────────────
@app.post("/predict/demand")
def predict_demand(req: DemandRequest):
    rolling_3h   = round((req.lag_1h + req.lag_2h) / 2)
    input_df     = pd.DataFrame([{
        'pickup_cluster': req.pickup_cluster,
        'hour'          : req.hour,
        'lag_1h'        : req.lag_1h,
        'lag_2h'        : req.lag_2h,
        'rolling_3h'    : rolling_3h
    }])
    pred         = max(0, int(demand_model.predict(input_df)[0]))
    level        = "High" if pred > 50 else "Medium" if pred > 25 else "Low"

    return {
        "predicted_demand" : pred,
        "demand_level"     : level,
        "zone"             : req.pickup_cluster,
        "hour"             : req.hour
    }

# ── 2. Supply Prediction ──────────────────────────────────────
@app.post("/predict/supply")
def predict_supply(req: SupplyRequest):
    supply_trend = req.supply_lag_1h - req.supply_lag_2h
    rolling_3h   = round((req.supply_lag_1h + req.supply_lag_2h) / 2)
    input_df     = pd.DataFrame([{
        'hour'               : req.hour,
        'day_of_week'        : req.day_of_week,
        'is_weekend'         : req.is_weekend,
        'cluster_avg_supply' : req.cluster_avg_supply,
        'supply_lag_1h'      : req.supply_lag_1h,
        'supply_lag_2h'      : req.supply_lag_2h,
        'supply_trend'       : supply_trend,
        'rolling_3h'         : rolling_3h,
        'traffic_level_enc'  : req.traffic_level_enc
    }])
    pred         = max(0, int(supply_model.predict(input_df)[0]))
    status       = "Good" if pred > 30 else "Moderate" if pred > 15 else "Shortage"

    return {
        "predicted_supply" : pred,
        "supply_status"    : status,
        "hour"             : req.hour
    }

# ── 3. Cancellation Prediction ────────────────────────────────
@app.post("/predict/cancel")
def predict_cancel(req: CancelRequest):
    input_df     = pd.DataFrame([{
        'hour'             : req.hour,
        'day_of_week'      : req.day_of_week,
        'is_weekend'       : req.is_weekend,
        'traffic_level_enc': req.traffic_level_enc,
        'gap'              : req.gap,
        'surge_multiplier' : req.surge_multiplier,
        'pickup_cluster'   : req.pickup_cluster
    }])
    prob         = float(cancel_model.predict_proba(input_df)[0][1])
    pred         = int(cancel_model.predict(input_df)[0])
    risk         = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    return {
        "cancel_probability" : round(prob, 4),
        "prediction"         : "Cancelled" if pred == 1 else "Completed",
        "risk_level"         : risk
    }

# ── 4. Gap & Dynamic Pricing ──────────────────────────────────
@app.post("/predict/gap")
def predict_gap(req: DemandRequest):
    # Get demand
    rolling_3h   = round((req.lag_1h + req.lag_2h) / 2)
    demand_df    = pd.DataFrame([{
        'pickup_cluster': req.pickup_cluster,
        'hour'          : req.hour,
        'lag_1h'        : req.lag_1h,
        'lag_2h'        : req.lag_2h,
        'rolling_3h'    : rolling_3h
    }])
    pred_demand  = max(0, int(demand_model.predict(demand_df)[0]))

    # Estimate supply
    sup_lag_1h   = int(pred_demand * 0.7)
    sup_lag_2h   = int(pred_demand * 0.65)
    supply_df    = pd.DataFrame([{
        'hour'               : req.hour,
        'day_of_week'        : 0,
        'is_weekend'         : 0,
        'cluster_avg_supply' : 25,
        'supply_lag_1h'      : sup_lag_1h,
        'supply_lag_2h'      : sup_lag_2h,
        'supply_trend'       : sup_lag_1h - sup_lag_2h,
        'rolling_3h'         : round((sup_lag_1h+sup_lag_2h)/2),
        'traffic_level_enc'  : 1
    }])
    pred_supply  = max(0, int(supply_model.predict(supply_df)[0]))

    gap          = pred_demand - pred_supply
    surge        = calc_surge(gap)
    base_fare    = 200
    dynamic_fare = round(base_fare * surge, 2)
    zone_status  = "High Demand" if surge >= 2.0 else \
                   "Moderate"    if surge >= 1.3 else "Normal"

    return {
        "predicted_demand"  : pred_demand,
        "predicted_supply"  : pred_supply,
        "gap"               : gap,
        "surge_multiplier"  : surge,
        "base_fare"         : base_fare,
        "dynamic_fare"      : dynamic_fare,
        "zone_status"       : zone_status
    }

# ── 5. Feedback Analysis ──────────────────────────────────────
@app.post("/analyze/feedback")
def analyze_feedback(req: FeedbackRequest):
    prompt   = f"""Analyze this ride-hailing feedback. Respond ONLY in this format:
Sentiment: [Positive/Neutral/Negative]
Issue: [Driver Behavior/Vehicle Condition/Cancellation/Delay/Route & Fare/General/None]
Score: [1-10]
Summary: [One sentence]

Feedback: "{req.feedback_text}"
"""
    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        max_tokens  = 100,
        temperature = 0
    )
    result    = response.choices[0].message.content.strip()
    lines     = result.split('\n')
    parsed    = {"sentiment":"Neutral","issue":"General","score":5,"summary":""}

    for line in lines:
        if line.startswith('Sentiment:'):
            parsed['sentiment'] = line.split(':')[1].strip()
        elif line.startswith('Issue:'):
            parsed['issue']     = line.split(':')[1].strip()
        elif line.startswith('Score:'):
            try: parsed['score'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('Summary:'):
            parsed['summary']   = line.split(':',1)[1].strip()

    return {
        "feedback"  : req.feedback_text,
        "sentiment" : parsed['sentiment'],
        "issue"     : parsed['issue'],
        "score"     : parsed['score'],
        "summary"   : parsed['summary']
    }

# ── 6. Driver Matching ────────────────────────────────────────
@app.post("/match/driver")
def match_driver(req: MatchRequest):
    filtered = driver_profile[
        (driver_profile['avg_rating']        >= req.min_rating) &
        (driver_profile['avg_eta']           <= req.max_eta)    &
        (driver_profile['cancellation_rate'] <= req.max_cancel_rate)
    ].nlargest(3, 'match_score')

    if filtered.empty:
        return {"error": "No drivers match criteria"}

    best    = filtered.iloc[0]
    top3    = filtered[['driver_id','avg_rating','avg_eta',
                         'cancellation_rate','match_score']].to_dict(orient='records')

    return {
        "recommended_driver" : {
            "driver_id"        : int(best['driver_id']),
            "avg_rating"       : round(best['avg_rating'], 3),
            "avg_eta"          : round(best['avg_eta'],    2),
            "cancellation_rate": round(best['cancellation_rate'], 3),
            "match_score"      : round(best['match_score'], 4)
        },
        "top_3_drivers"      : top3,
        "total_matches"      : len(driver_profile[
            (driver_profile['avg_rating']        >= req.min_rating) &
            (driver_profile['avg_eta']           <= req.max_eta)    &
            (driver_profile['cancellation_rate'] <= req.max_cancel_rate)
        ])
    }

# ── 7. AI Chatbot ─────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    system_prompt = f"""You are RideFlow AI Assistant in Chennai.
Talking to a {req.role}. Respond in {req.language}.
Be concise — max 3 sentences. Be professional and empathetic.
Refunds: 3-5 business days."""

    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user",  "content":req.message}
        ],
        max_tokens  = 150,
        temperature = 0.5
    )

    return {
        "message"  : req.message,
        "response" : response.choices[0].message.content.strip(),
        "language" : req.language,
        "role"     : req.role
    }
