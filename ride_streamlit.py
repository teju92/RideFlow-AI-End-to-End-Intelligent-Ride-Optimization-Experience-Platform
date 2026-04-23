import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import joblib
import json
import time
from groq import Groq

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "RideFlow AI Platform",
    page_icon  = "🚗",
    layout     = "wide"
)

# ── Groq client ───────────────────────────────────────────────
GROQ_API_KEY = "Your Key"  # 👈 add your key
client = Groq(api_key=GROQ_API_KEY)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("rideflow_module3_groq.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'trip_distance_km' not in df.columns:
        df['trip_distance_km'] = df.apply(
            lambda row: geodesic(
                (row['pickup_lat'], row['pickup_long']),
                (row['drop_lat'],   row['drop_long'])
            ).km, axis=1
        )
    return df

@st.cache_data
def load_driver_profile():
    return joblib.load('driver_profile.pkl')

@st.cache_data
def load_feedback_results():
    with open('feedback_results.json', 'r') as f:
        return json.load(f)

df             = load_data()
driver_profile = load_driver_profile()
feedback_map   = load_feedback_results()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/car--v1.png", width=80)
st.sidebar.title("RideFlow AI")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 EDA & Insights",
    "🤖 Feedback Intelligence",
    "🚗 Ride Matching",
    "💬 AI Chatbot",
    "📈 ML Predictions"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Rides:** {len(df):,}")
st.sidebar.markdown(f"**Date Range:** {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
st.sidebar.markdown(f"**Total Drivers:** {df['driver_id'].nunique():,}")

# ═══════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚗 RideFlow AI Platform")
    st.markdown("### End-to-End Intelligent Ride Optimization — Chennai")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rides", f"{len(df):,}")
    with col2:
        completed = (df['ride_status'] == 'completed').sum()
        st.metric("Completed", f"{completed:,}",
                  delta=f"{completed/len(df)*100:.1f}%")
    with col3:
        st.metric("Avg Fare", f"₹{df['fare_price'].mean():.0f}")
    with col4:
        st.metric("Avg Rating", f"{df['driver_rating'].mean():.2f} ⭐")

    st.markdown("---")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Avg Distance", f"{df['trip_distance_km'].mean():.1f} km")
    with col6:
        st.metric("Avg ETA", f"{df['actual_eta_min'].mean():.1f} min")
    with col7:
        cancelled = (df['ride_status'] == 'cancelled').sum()
        st.metric("Cancellations", f"{cancelled:,}",
                  delta=f"-{cancelled/len(df)*100:.1f}%",
                  delta_color="inverse")
    with col8:
        st.metric("Total Revenue", f"₹{df['fare_price'].sum()/1e6:.2f}M")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Daily Ride Volume")
        rides_by_date = df.groupby(df['timestamp'].dt.date).size().reset_index()
        rides_by_date.columns = ['date', 'rides']
        fig = px.line(rides_by_date, x='date', y='rides')
        fig.update_traces(line_color='#2196F3')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🔵 Ride Status Split")
        fig2 = px.pie(df, names='ride_status',
                      color_discrete_sequence=['#4CAF50','#F44336'],
                      hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🌤️ Rides by Weather")
        weather_counts = df['weather'].value_counts().reset_index()
        weather_counts.columns = ['weather', 'count']
        fig3 = px.bar(weather_counts, x='weather', y='count', color='weather')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("🚦 Rides by Traffic")
        traffic_counts = df['traffic_level'].value_counts().reset_index()
        traffic_counts.columns = ['traffic', 'count']
        fig4 = px.bar(traffic_counts, x='traffic', y='count',
                      color='traffic',
                      color_discrete_map={
                          'low':'#4CAF50','medium':'#FF9800','high':'#F44336'})
        st.plotly_chart(fig4, use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# 📊 EDA & INSIGHTS
# ═══════════════════════════════════════════════════════════════
if page == "📊 EDA & Insights":
    st.title("📊 EDA & Insights")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⏱️ Estimated vs Actual ETA")
        fig = px.scatter(df.sample(2000),
                         x='estimated_eta_min', y='actual_eta_min',
                         color='traffic_level',
                         color_discrete_map={
                             'low':'#4CAF50','medium':'#FF9800','high':'#F44336'},
                         opacity=0.5)
        fig.add_shape(type='line', x0=0, y0=0, x1=20, y1=20,
                      line=dict(color='blue', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📦 Trip Distance Distribution")
        fig = px.histogram(df, x='trip_distance_km', nbins=50,
                           color_discrete_sequence=['#2196F3'])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📍 Top 10 Pickup Zones")
        top_zones = df['pickup_zone'].value_counts().head(10).reset_index()
        top_zones.columns = ['zone', 'count']
        fig = px.bar(top_zones, x='count', y='zone',
                     orientation='h', color='count',
                     color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("🕐 Rides by Hour of Day")
        hourly = df.groupby('hour').size().reset_index()
        hourly.columns = ['hour', 'rides']
        fig = px.bar(hourly, x='hour', y='rides',
                     color='rides', color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("💰 Fare vs Trip Distance")
        fig = px.scatter(df.sample(2000),
                         x='trip_distance_km', y='fare_price',
                         color='surge_multiplier',
                         opacity=0.5,
                         color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("⭐ Driver Rating Distribution")
        fig = px.histogram(df, x='driver_rating', nbins=30,
                           color_discrete_sequence=['#9C27B0'])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col7, col8 = st.columns(2)
    with col7:
        st.subheader("🌦️ Fare by Weather")
        fig = px.box(df, x='weather', y='fare_price', color='weather')
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        st.subheader("📅 Rides by Day of Week")
        dow = df.groupby('day_of_week').size().reset_index()
        dow.columns = ['day', 'rides']
        dow['day_name'] = dow['day'].map({
            0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
        fig = px.bar(dow, x='day_name', y='rides',
                     color='rides', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 🤖 FEEDBACK INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
if page == "🤖 Feedback Intelligence":
    st.title("🤖 Feedback Intelligence System")
    st.markdown("### Groq LLM — Sentiment & Issue Classification")
    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    sentiment_counts = df['groq_sentiment'].value_counts()

    with col1:
        st.metric("😊 Positive", f"{sentiment_counts.get('Positive',0):,}",
                  delta=f"{sentiment_counts.get('Positive',0)/len(df)*100:.1f}%")
    with col2:
        st.metric("😐 Neutral", f"{sentiment_counts.get('Neutral',0):,}",
                  delta=f"{sentiment_counts.get('Neutral',0)/len(df)*100:.1f}%")
    with col3:
        st.metric("😠 Negative", f"{sentiment_counts.get('Negative',0):,}",
                  delta=f"-{sentiment_counts.get('Negative',0)/len(df)*100:.1f}%",
                  delta_color="inverse")
    with col4:
        st.metric("Avg Score", f"{df['groq_score'].mean():.1f}/10")

    st.markdown("---")

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("📊 Sentiment Distribution")
        fig = px.pie(df, names='groq_sentiment',
                     color='groq_sentiment',
                     color_discrete_map={
                         'Positive':'#4CAF50',
                         'Neutral' :'#FF9800',
                         'Negative':'#F44336'},
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader("🏷️ Issue Category Breakdown")
        issue_counts = df['groq_issue'].value_counts().reset_index()
        issue_counts.columns = ['issue', 'count']
        fig = px.bar(issue_counts, x='issue', y='count',
                     color='issue')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col6, col7 = st.columns(2)
    with col6:
        st.subheader("📍 Sentiment by Pickup Zone (Top 8)")
        top_zones = df['pickup_zone'].value_counts().head(8).index
        zone_sent = df[df['pickup_zone'].isin(top_zones)].groupby(
            ['pickup_zone','groq_sentiment']).size().reset_index(name='count')
        fig = px.bar(zone_sent, x='pickup_zone', y='count',
                     color='groq_sentiment', barmode='group',
                     color_discrete_map={
                         'Positive':'#4CAF50',
                         'Neutral' :'#FF9800',
                         'Negative':'#F44336'})
        st.plotly_chart(fig, use_container_width=True)

    with col7:
        st.subheader("⭐ Satisfaction Score by Issue")
        score_issue = df.groupby('groq_issue')['groq_score'].mean().reset_index()
        score_issue.columns = ['issue', 'avg_score']
        fig = px.bar(score_issue, x='issue', y='avg_score',
                     color='avg_score',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Live Groq Analyzer ────────────────────────────────────
    st.subheader("🔍 Live Feedback Analyzer — Powered by Groq")

    user_feedback = st.text_area(
        "Enter customer feedback:",
        placeholder="e.g. Driver was very rude and cancelled at the last minute...",
        height=100
    )

    if st.button("🔍 Analyze Feedback"):
        if user_feedback:
            with st.spinner("Analyzing with Groq LLM..."):
                prompt = f"""Analyze this ride-hailing feedback and respond ONLY in this exact format:
Sentiment: [Positive/Neutral/Negative]
Issue: [Driver Behavior/Vehicle Condition/Cancellation/Delay/Route & Fare/General/None]
Score: [1-10]
Summary: [One sentence explanation]

Feedback: "{user_feedback}"
"""
                response = client.chat.completions.create(
                    model    = "llama-3.3-70b-versatile",
                    messages = [{"role":"user","content":prompt}],
                    max_tokens  = 100,
                    temperature = 0
                )
                result = response.choices[0].message.content.strip()
                lines  = result.split('\n')

                sentiment = "Neutral"
                issue     = "General"
                score     = 5
                summary   = ""

                for line in lines:
                    if line.startswith('Sentiment:'):
                        sentiment = line.split(':')[1].strip()
                    elif line.startswith('Issue:'):
                        issue = line.split(':')[1].strip()
                    elif line.startswith('Score:'):
                        try: score = int(line.split(':')[1].strip())
                        except: score = 5
                    elif line.startswith('Summary:'):
                        summary = line.split(':')[1].strip()

                color_map = {'Positive':'green','Neutral':'orange','Negative':'red'}
                color     = color_map.get(sentiment, 'gray')

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Sentiment", sentiment)
                with col_r2:
                    st.metric("Issue Category", issue)
                with col_r3:
                    st.metric("Score", f"{score}/10")

                st.info(f"💡 {summary}")
        else:
            st.warning("Please enter feedback text!")


# ═══════════════════════════════════════════════════════════════
# 🚗 RIDE MATCHING
# ═══════════════════════════════════════════════════════════════
if page == "🚗 Ride Matching":
    st.title("🚗 AI Ride Matching Assistant")
    st.markdown("### Groq LLM — Smart Driver Recommendation")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────
    st.subheader("🎛️ Ride Request Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        location = st.selectbox("📍 Pickup Zone", 
            ['T Nagar','Anna Nagar','Tambaram','Velachery',
             'Adyar','Porur','OMR','Chennai Central','Chennai Airport'])
    with col2:
        hour = st.slider("🕐 Hour of Day", 0, 23, 18)
    with col3:
        traffic = st.selectbox("🚦 Traffic Level", ['low','medium','high'])

    col4, col5, col6 = st.columns(3)
    with col4:
        min_rating = st.slider("⭐ Min Driver Rating", 2.0, 5.0, 3.5, 0.1)
    with col5:
        max_eta = st.slider("⏱️ Max ETA (mins)", 5, 25, 20)
    with col6:
        max_cancel = st.slider("❌ Max Cancel Rate", 0.0, 1.0, 0.4, 0.05)

    st.markdown("---")

    # ── Filter drivers ────────────────────────────────────────
    filtered = driver_profile[
        (driver_profile['avg_rating']        >= min_rating) &
        (driver_profile['avg_eta']           <= max_eta)    &
        (driver_profile['cancellation_rate'] <= max_cancel)
    ].nlargest(10, 'match_score')

    st.subheader(f"✅ {len(filtered)} Drivers Match Your Criteria")

    if len(filtered) == 0:
        st.warning("⚠️ No drivers match. Try relaxing the filters.")
    else:
        # ── Top 3 Cards ───────────────────────────────────────
        st.subheader("🏆 Top 3 Recommended Drivers")
        top3   = filtered.head(3)
        medals = ["🥇","🥈","🥉"]
        cols   = st.columns(3)

        for i, (_, driver) in enumerate(top3.iterrows()):
            with cols[i]:
                cr_color = "🟢" if driver['cancellation_rate'] < 0.2 else \
                           "🟡" if driver['cancellation_rate'] < 0.3 else "🔴"
                st.markdown(f"""
                <div style="background:#f0f2f6;padding:20px;
                            border-radius:12px;text-align:center;">
                    <h2>{medals[i]}</h2>
                    <h3>Driver #{int(driver['driver_id'])}</h3>
                    <hr>
                    <p>⭐ Rating: <b>{driver['avg_rating']:.2f}</b></p>
                    <p>⏱️ ETA: <b>{driver['avg_eta']:.1f} mins</b></p>
                    <p>{cr_color} Cancel Rate: <b>{driver['cancellation_rate']*100:.1f}%</b></p>
                    <p>🎯 Match Score: <b>{driver['match_score']:.4f}</b></p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Groq Recommendation ───────────────────────────────
        st.subheader("🤖 Groq AI Recommendation")

        if st.button("🚀 Get AI Recommendation"):
            with st.spinner("Groq is analyzing drivers..."):
                drivers_info = top3[
                    ['driver_id','avg_rating','avg_eta',
                     'cancellation_rate','match_score']
                ].to_dict(orient='records')

                prompt = f"""You are an AI Ride Matching Assistant for RideFlow in Chennai.

Customer Location: {location}
Time: {hour}:00 hrs
Traffic: {traffic}

Top 3 Available Drivers:
{json.dumps(drivers_info, indent=2)}

Field info:
- avg_rating: out of 5 (higher=better)
- avg_eta: minutes to arrive (lower=better)
- cancellation_rate: 0-1 (lower=better)
- match_score: 0-1 (higher=better)

Respond in EXACTLY this format:
Recommended Driver ID: [ID]
Match Score: [score]
Reason: [2-3 sentences on ETA, rating, cancellation risk]
Alternative: Driver [ID] is also strong because [one sentence]
Risk Flag: [warning if cancel_rate>0.25, else None]
Confidence: [High/Medium/Low]"""

                response = client.chat.completions.create(
                    model    = "llama-3.3-70b-versatile",
                    messages = [{"role":"user","content":prompt}],
                    max_tokens  = 300,
                    temperature = 0.3
                )
                result = response.choices[0].message.content.strip()
                st.success(result)

        st.markdown("---")

        # ── Comparison Chart ──────────────────────────────────
        st.subheader("📊 Driver Comparison")
        fig = px.bar(filtered, x='driver_id', y='match_score',
                     color='match_score', color_continuous_scale='Blues',
                     hover_data=['avg_rating','avg_eta','cancellation_rate'])
        st.plotly_chart(fig, use_container_width=True)

        # ── Full Table ────────────────────────────────────────
        st.subheader("📋 Driver Shortlist")
        st.dataframe(
            filtered[['driver_id','avg_rating','avg_eta',
                      'cancellation_rate','total_rides','match_score']]
            .reset_index(drop=True),
            use_container_width=True
        )




# ═══════════════════════════════════════════════════════════════
# 💬 AI CHATBOT
# ═══════════════════════════════════════════════════════════════
if page == "💬 AI Chatbot":
    st.title("💬 AI Chatbot & Multilingual Support")
    st.markdown("### Groq LLM — Customer & Driver Support")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox("🌐 Language", ["English","Tamil","Hindi"])
    with col2:
        role = st.selectbox("👤 I am a", ["Customer","Driver"])

    st.markdown("---")

    # ── Session state init ────────────────────────────────────
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'groq_history' not in st.session_state:
        st.session_state['groq_history'] = []
    if 'prev_lang' not in st.session_state:
        st.session_state['prev_lang'] = language
    if 'prev_role' not in st.session_state:
        st.session_state['prev_role'] = role

    # Reset on language/role change
    if (st.session_state['prev_lang'] != language or
        st.session_state['prev_role'] != role):
        st.session_state['chat_history'] = []
        st.session_state['groq_history'] = []
        st.session_state['prev_lang']    = language
        st.session_state['prev_role']    = role

    system_prompt = f"""You are RideFlow AI Assistant for a ride-hailing platform in Chennai.
You are talking to a {role}.
Respond ONLY in {language}. Be concise — max 3 sentences.
Customer help: booking, status, refunds, ETA, complaints.
Driver help: earnings, navigation, ratings, support.
Refunds take 3-5 business days. Never make up specific ride details."""

    def get_groq_reply(user_msg):
        messages = [{"role":"system","content":system_prompt}]
        messages.extend(st.session_state['groq_history'])
        messages.append({"role":"user","content":user_msg})
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            max_tokens  = 150,
            temperature = 0.5
        )
        return response.choices[0].message.content.strip()

    def send_message(user_msg):
        reply = get_groq_reply(user_msg)
        st.session_state['chat_history'].append({"role":"user",      "content":user_msg})
        st.session_state['chat_history'].append({"role":"assistant", "content":reply})
        st.session_state['groq_history'].append({"role":"user",      "content":user_msg})
        st.session_state['groq_history'].append({"role":"assistant", "content":reply})

    # ── Quick buttons ─────────────────────────────────────────
    st.markdown("**⚡ Quick Questions:**")
    quick_questions = {
        "English": {
            "Customer": ["Where is my driver?","Cancel my ride",
                         "I want a refund","My ride was delayed"],
            "Driver"  : ["What are my earnings?","Why did my rating drop?",
                         "Navigate to OMR","Cancel a ride"]
        },
        "Tamil": {
            "Customer": ["என் டிரைவர் எங்கே?","பயணம் ரத்து செய்",
                         "பணம் திரும்ப வேண்டும்","தாமதமான பயணம்"],
            "Driver"  : ["இன்று வருமானம்?","ரேட்டிங் குறைந்தது ஏன்?",
                         "OMR வழி காட்டு","பயணம் ரத்து"]
        },
        "Hindi": {
            "Customer": ["मेरा ड्राइवर कहाँ?","राइड रद्द करो",
                         "रिफंड चाहिए","राइड देरी से आई"],
            "Driver"  : ["आज की कमाई?","रेटिंग क्यों गिरी?",
                         "OMR कैसे जाएं?","राइड रद्द करनी है"]
        }
    }

    questions = quick_questions[language][role]
    q_cols    = st.columns(4)
    for i, (col, question) in enumerate(zip(q_cols, questions)):
        with col:
            if st.button(question, key=f"quick_{i}"):
                with st.spinner("Typing..."):
                    send_message(question)
                st.rerun()

    st.markdown("---")

    # ── Chat display ──────────────────────────────────────────
    st.subheader(f"💬 {language} | {role} Support")
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # ── Chat input ────────────────────────────────────────────
    if prompt := st.chat_input("Type your message here..."):
        with st.spinner("RideFlow AI is typing..."):
            send_message(prompt)
        st.rerun()

    # ── Clear ─────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat"):
        st.session_state['chat_history'] = []
        st.session_state['groq_history'] = []
        st.rerun()




# ═══════════════════════════════════════════════════════════════
# 📈 ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════
if page == "📈 ML Predictions":
    st.title("📈 ML Predictions — Module 1")
    st.markdown("### Demand | Supply | Cancellation Risk")
    st.markdown("---")

    @st.cache_resource
    def load_models():
        return {
            'demand' : joblib.load('demand_prediction_model.pkl'),
            'supply' : joblib.load('driver_supply_model.pkl'),
            'cancel' : joblib.load('cancellation_model.pkl'),
        }

    models = load_models()

    # ── Input Parameters ──────────────────────────────────────
    st.subheader("🎛️ Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        hour        = st.slider("🕐 Hour of Day",    0, 23, 18)
        day_of_week = st.selectbox("📅 Day",
                        ["Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday"])
        dow_enc     = ["Monday","Tuesday","Wednesday","Thursday",
                       "Friday","Saturday","Sunday"].index(day_of_week)
        is_weekend  = 1 if day_of_week in ["Saturday","Sunday"] else 0

    with col2:
        cluster     = st.slider("📍 Zone Cluster",    0, 9,   3)
        traffic     = st.selectbox("🚦 Traffic",["low","medium","high"])
        traffic_enc = {"low":0,"medium":1,"high":2}[traffic]

    with col3:
        surge       = st.slider("⚡ Surge Multiplier",1.0, 3.0, 1.2, 0.1)
        lag_1h      = st.slider("📊 Last Hour Demand", 5,  100, 45)
        lag_2h      = st.slider("📊 2hr Ago Demand",   5,  100, 40)

    st.markdown("---")

    if st.button("🚀 Run Predictions"):

        col_r1, col_r2, col_r3 = st.columns(3)

        # ── 1️⃣ Demand ────────────────────────────────────────
        with col_r1:
            st.subheader("1️⃣ Ride Demand")
            try:
                # Features: pickup_cluster, hour, lag_1h, lag_2h, rolling_3h
                rolling_3h   = round((lag_1h + lag_2h) / 2)
                demand_input = pd.DataFrame([{
                    'pickup_cluster': cluster,
                    'hour'          : hour,
                    'lag_1h'        : lag_1h,
                    'lag_2h'        : lag_2h,
                    'rolling_3h'    : rolling_3h
                }])
                pred_demand = max(0, int(models['demand'].predict(demand_input)[0]))
                st.metric("Predicted Rides", f"{pred_demand}")
                status = "🔴 High"   if pred_demand > 50 else \
                         "🟡 Medium" if pred_demand > 25 else "🟢 Low"
                st.markdown(f"**Demand Level:** {status}")
                st.success(f"Zone {cluster} | {hour}:00hrs → {pred_demand} rides")
            except Exception as e:
                st.error(f"Demand error: {e}")

        # ── 2️⃣ Supply ────────────────────────────────────────
        with col_r2:
            st.subheader("2️⃣ Driver Supply")
            try:
                # Features: hour, day_of_week, is_weekend, cluster_avg_supply,
                #           supply_lag_1h, supply_lag_2h, supply_trend,
                #           rolling_3h, traffic_level_enc
                sup_lag_1h   = int(pred_demand * 0.7)
                sup_lag_2h   = int(pred_demand * 0.65)
                supply_trend = sup_lag_1h - sup_lag_2h
                supply_input = pd.DataFrame([{
                    'hour'               : hour,
                    'day_of_week'        : dow_enc,
                    'is_weekend'         : is_weekend,
                    'cluster_avg_supply' : 25,
                    'supply_lag_1h'      : sup_lag_1h,
                    'supply_lag_2h'      : sup_lag_2h,
                    'supply_trend'       : supply_trend,
                    'rolling_3h'         : round((sup_lag_1h+sup_lag_2h)/2),
                    'traffic_level_enc'  : traffic_enc
                }])
                pred_supply = max(0, int(models['supply'].predict(supply_input)[0]))
                st.metric("Available Drivers", f"{pred_supply}")
                status = "🟢 Good"     if pred_supply > 30 else \
                         "🟡 Moderate" if pred_supply > 15 else "🔴 Shortage"
                st.markdown(f"**Supply Status:** {status}")
                st.success(f"Zone {cluster} | {hour}:00hrs → {pred_supply} drivers")
            except Exception as e:
                st.error(f"Supply error: {e}")

        # ── 3️⃣ Gap & Surge ───────────────────────────────────
        with col_r3:
            st.subheader("3️⃣ Demand-Supply Gap")
            try:
                gap = pred_demand - pred_supply

                def calc_surge(gap):
                    if gap <= 0:    return 1.0
                    elif gap <= 5:  return round(1.0 + (gap/5)*0.5, 2)
                    elif gap <= 15: return round(1.5 + (gap-5)/10,  2)
                    else:           return 3.0

                surge_val    = calc_surge(gap)
                base_fare    = 200
                dynamic_fare = round(base_fare * surge_val, 2)

                st.metric("Gap",            f"{gap} rides")
                st.metric("Surge",          f"{surge_val}x")
                st.metric("Dynamic Fare",   f"₹{dynamic_fare}")
                zone = "🔴 High Demand" if surge_val >= 2.0 else \
                       "🟡 Moderate"    if surge_val >= 1.3 else \
                       "🟢 Normal"
                st.markdown(f"**Zone:** {zone}")
            except Exception as e:
                st.error(f"Gap error: {e}")

        st.markdown("---")

        # ── 4️⃣ Cancellation ──────────────────────────────────
        st.subheader("4️⃣ Cancellation Risk Prediction")
        try:
            # Features: hour, day_of_week, is_weekend, traffic_level_enc,
            #           gap, surge_multiplier, pickup_cluster
            cancel_input = pd.DataFrame([{
                'hour'             : hour,
                'day_of_week'      : dow_enc,
                'is_weekend'       : is_weekend,
                'traffic_level_enc': traffic_enc,
                'gap'              : gap,
                'surge_multiplier' : surge_val,
                'pickup_cluster'   : cluster
            }])

            cancel_prob = models['cancel'].predict_proba(cancel_input)[0][1]
            cancel_pred = models['cancel'].predict(cancel_input)[0]

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.metric("Cancel Probability", f"{cancel_prob*100:.1f}%")
            with col_c2:
                pred_label = "🔴 Likely Cancel" if cancel_pred == 1 \
                             else "🟢 Likely Complete"
                st.metric("Prediction", pred_label)
            with col_c3:
                risk  = "High"   if cancel_prob > 0.6 else \
                        "Medium" if cancel_prob > 0.3 else "Low"
                emoji = "⚠️" if risk != "Low" else "✅"
                st.metric("Risk Level", f"{emoji} {risk}")

            st.markdown("**Cancellation Probability:**")
            st.progress(float(cancel_prob))

        except Exception as e:
            st.error(f"Cancel error: {e}")

        st.markdown("---")

        # ── Summary ───────────────────────────────────────────
        st.subheader("📋 Prediction Summary")
        try:
            st.info(f"""
**Zone {cluster} | {day_of_week} {hour}:00 | Traffic: {traffic.upper()}**

🚗 Predicted Demand  : {pred_demand} rides
🧑 Available Drivers : {pred_supply} drivers
📊 Gap               : {gap} rides
⚡ Surge Multiplier  : {surge_val}x
💰 Dynamic Fare      : ₹{base_fare} → ₹{dynamic_fare}
🚫 Cancel Risk       : {cancel_prob*100:.1f}%
            """)
        except:
            pass