# News Summarization, Sentiment Analysis & Text-to-Speech

This project extracts news articles, summarizes them, performs sentiment analysis, and converts summaries to Hindi speech.

## Features
✅ Fetches latest 10 news articles  
✅ Summarizes articles using LSA  
✅ Performs sentiment analysis (Positive, Negative, Neutral)  
✅ Converts summaries to speech in Hindi  
✅ Displays sentiment distribution  

## Setup Instructions
### 1️⃣ Install Dependencies

pip install -r requirements.txt

### 2️⃣ Run the FastAPI Backend

uvicorn api:app --reload

### 3️⃣ Run the Streamlit Frontend

streamlit run app.py

### 4️⃣ Open in Browser
Go to `http://localhost:8501/`

---
## API Endpoints
### 1️⃣ **Analyze News (`GET /analyze?company=Tesla`)**
🔹 Returns JSON with article titles, summaries, and sentiment analysis.
