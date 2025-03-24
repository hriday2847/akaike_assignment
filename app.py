import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"  

st.title("📢 AI-Powered News Sentiment Analyzer")

# ✅ User Input for Company Name
company_name = st.text_input("Enter Company Name:", "Tesla")

if st.button("Fetch News"):
    response = requests.get(f"{API_URL}/news/{company_name}")

    if response.status_code == 200:
        data = response.json()
        
        st.subheader(f"📊 Sentiment Distribution for {company_name}")
        st.bar_chart(data["sentiment_distribution"])

        for idx, article in enumerate(data["articles"]):
            st.subheader(f"{idx+1}. {article['title']}")
            st.write(f"📅 Published: {article['publish_date']}")
            st.write(f"🔗 [Read More]({article['source']})")
            st.write(f"📢 Sentiment: **{article['sentiment']}**")
            st.write("📖 Summary:", article["summary"])

            # ✅ TTS Button
            if st.button(f"🔊 Listen to Summary {idx+1}"):
                tts_response = requests.get(f"{API_URL}/tts/{article['summary']}")
                if "file" in tts_response.json():
                    st.audio(tts_response.json()["file"], format="audio/mp3")
    else:
        st.error("Error fetching news!")