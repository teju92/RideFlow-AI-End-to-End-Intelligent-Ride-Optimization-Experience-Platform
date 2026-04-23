**🚗 RideFlow AI: End-to-End Intelligent Ride Optimization Platform**


RideFlow AI is an advanced, multi-module AI system designed to optimize ride-hailing operations in Chennai. It balances demand and driver supply, reduces cancellations, and enhances customer experience through a layered architecture of Machine Learning, Deep Learning, and Generative AI.


**📌 Problem Statement**


Traditional ride-hailing systems often suffer from high cancellation rates, inaccurate ETAs, and inefficient driver distribution. This project aims to:

Balance Demand & Supply: Predict ride volume and driver availability per zone.

Reduce Cancellations: Predict high-risk rides before they fail.

Optimize Pricing: Dynamically adjust fares based on real-time demand-supply gaps.

Enhance Experience: Provide AI-powered support and feedback analysis.




**🏗️ System Architecture & Pipeline**


The project follows a modular 5-phase pipeline to ensure scalability and maintainability:

Data Foundation: Geospatial encoding (K-Means clustering for zones), time-feature extraction, and data cleaning.

Core ML Layer: Predicts demand (XGBoost/Random Forest), supply, and cancellation probability.

NLP & Decision Layer: Uses BERT for sentiment analysis of feedback and LLMs (Llama 3 via Groq) for matching and support.

Interaction Layer: Streamlit Dashboard and FastAPI endpoints for real-time deployment.

**🛠️ Tech Stack & Frameworks**


Data Science: Python, Pandas, NumPy, Scikit-Learn.

Machine Learning: XGBoost, Joblib (for model persistence).

NLP/GenAI: BERT (Feedback Analysis), Groq API (Llama-3.3-70b-versatile).

Deployment: FastAPI (Backend API), Streamlit (Frontend Dashboard).

Geospatial: Geopy, K-Means Clustering.

**📊 Modules & Results**


**1. Machine Learning (Core Intelligence)**
Demand & Supply Prediction: Predicts rides and drivers per zone.

Result: Achieved R² > 0.75 for demand and MAE < 10 for supply.

Cancellation Prediction: Uses features like ETA, surge multiplier, and driver rating.

Result: Accuracy: 81.3% | F1-score: 0.76 (Exceeding project benchmarks).

**2. Feedback & Support (NLP)**
Sentiment Analysis: Categorizes feedback (e.g., "Driver was rude" → Negative).
AI Chatbot: A multilingual assistant (Tamil, Hindi, English) for instant support.

**💡 Why This Approach?**


K-Means for Geospatial Zones: Instead of raw Lat/Long, we cluster coordinates into "Zones." This allows the models to learn local demand patterns more effectively.

Modular API Design: By using FastAPI, each model (Demand, Cancel, Supply) acts as a microservice, making it easy to update the cancellation model without affecting the pricing logic.

LLM Integration: We chose Llama-3 (Groq) for the matching assistant because it provides "Explainable AI"—it doesn't just recommend a driver; it explains why (e.g., "Low cancellation risk + High rating").

**🚀 How to Run**


Clone the Repo: git clone <your-repo-link>

Install Dependencies: pip install -r requirements.txt

Run the API: uvicorn rideflow_api:app --reload

Run the Dashboard: streamlit run ride_streamlit.py


**⚠️ Limitations & Next Steps**


Data Limitations: The current model uses synthetic/historical data. Real-time traffic API integration (like Google Maps) would improve ETA accuracy.

Cold Start Problem: New zones with zero historical data currently rely on global averages.

Next Steps:

Implement LSTM for time-series demand forecasting to capture seasonal trends.

Deploy using Docker for consistent environment management.

Add Computer Vision (CNN) for driver behavior monitoring as per Module 2 specifications.


**📂 Folder Structure**
Plaintext
├── data/                  # Processed CSVs and feedback JSON
├── models/                # Saved .pkl files (XGBoost, K-Means, etc.)
├── notebooks/             # Module1_Final.ipynb, Ride_AI.ipynb
├── rideflow_api.py        # FastAPI Backend
├── ride_streamlit.py      # Streamlit Frontend
├── API_test.ipynb         # Endpoint testing script
└── README.md              # Project documentation

