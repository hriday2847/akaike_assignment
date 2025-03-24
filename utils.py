import requests
import feedparser
from bs4 import BeautifulSoup
from transformers import pipeline
import spacy
import pandas as pd
from gtts import gTTS
import torch

nlp = spacy.load("en_core_web_sm") 

# ✅ Initialize models globally for efficiency
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def fetch_news_articles(company_name):
    API_KEY = "36bf84c73f2e4395823f0a5b6003ffdb"
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        print("🚨 NewsAPI request failed:", data)
        return []

    articles = []
    for article in data.get("articles", [])[:10]:  # Get max 10 articles
        articles.append({
            "title": article.get("title", "Title Not Available"),
            "link": article.get("url"),
            "published": article.get("publishedAt"),
            "content": article.get("description") or article.get("content", "")
        })

    return articles

# ✅ Extract article data
def extract_article_data(url):
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        
        if "text/html" not in response.headers.get("Content-Type", ""):
            print("🚨 Not a valid HTML page!")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        
        if not paragraphs:
            print("🚨 No paragraphs found in the article!")
            return ""

        return ' '.join([p.text.strip() for p in paragraphs])
    
    except requests.RequestException as e:
        print(f"🚨 Error fetching page: {e}")
        return ""

# ✅ Summarize extracted article
def summarize_article(url, max_length=66):
    article_text = extract_article_data(url)
    
    if not article_text.strip():
        return "No content found on the provided URL."
    
    if len(article_text) > 1024:
        article_text = article_text[:1024]  # BART has a 1024-token limit

    summary = summarizer(article_text, max_length=max_length, min_length=50, do_sample=False)
    
    return summary[0]['summary_text']

# ✅ Get sentiment of the text
def get_sentiment(text):
    result = sentiment_model(text)[0]

    label_mapping = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}
    
    return label_mapping.get(result["label"], "Neutral"), result["score"]

# ✅ Extract topics from text
def extract_topics(article_text):
    doc = nlp(article_text)
    topics = set()
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "EVENT"]:
            topics.add(ent.text)
    return list(topics)[:3]  # Return top 3 unique topics

# ✅ Compare two articles
def compare_articles(article_1, article_2):
    title_1 = article_1.get("title", "Title Not Available")
    title_2 = article_2.get("title", "Title Not Available")
    
    topics_1 = article_1.get("topics", [])
    topics_2 = article_2.get("topics", [])

    comparison_texts = [
        {
            "Comparison": f"Article 1 highlights {title_1}, while Article 2 discusses {title_2}.",
            "Impact": f"The first article focuses on {', '.join(topics_1)}, while the second covers {', '.join(topics_2)}."
        }
    ]

    topic_overlap = {
        "Common Topics": list(set(topics_1) & set(topics_2)),
        "Unique Topics in Article 1": list(set(topics_1) - set(topics_2)),
        "Unique Topics in Article 2": list(set(topics_2) - set(topics_1))
    }

    return comparison_texts, topic_overlap

# ✅ Analyze sentiment distribution
def analyze_sentiment_distribution(sentiments):
    if not sentiments:
        return {"Positive": 0, "Negative": 0, "Neutral": 0}
    df = pd.DataFrame(sentiments, columns=["Sentiment"])
    return df["Sentiment"].value_counts().to_dict()

# ✅ Convert summary to Hindi speech
def generate_hindi_audio(text, output_path="output.mp3"):
    tts = gTTS(text=text, lang='hi')
    tts.save(output_path)
    return output_path
