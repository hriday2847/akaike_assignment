from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from utils import fetch_news_articles,extract_article_data,summarize_article, get_sentiment, analyze_sentiment_distribution, generate_hindi_audio

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the News Analysis and TTS API!"}

@app.get("/news/{company}")
def get_news(company: str):
    articles = fetch_news_articles(company) or []  # Ensure it's a list
    results = []

    for article in articles:
        link = article.get("link")
        if not link:  
            print(f"Skipping article: No link found -> {article}")
            continue  # Skip articles without links
        
        metadata = {"summary": extract_article_data(link)}
        if not metadata:
            print(f"Skipping article: Failed to extract data -> {link}")
            continue  # Skip if extraction fails

        metadata["sentiment"] = get_sentiment(metadata["summary"])
        results.append(metadata)

    if not results:
        return {"articles": [], "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}}

    sentiment_distribution = analyze_sentiment_distribution([a["sentiment"] for a in results])

    return {"articles": results, "sentiment_distribution": sentiment_distribution}

@app.get("/tts/{company_name}")
def get_tts(company_name: str):
    news_data = get_news(company_name)
    summary_text = "\n".join([f"{art['title']}: {art['sentiment']}" for art in news_data['articles']])
    audio_path = generate_hindi_audio(summary_text)
    return {"audio_file": audio_path}