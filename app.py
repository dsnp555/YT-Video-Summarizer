import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from gtts import gTTS
from googletrans import Translator
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shutil

# Load API key from environment or streamlit secrets
load_dotenv()  # will not raise error if .env doesn't exist
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("No API key found. Please set GOOGLE_API_KEY in .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model
model = genai.GenerativeModel('gemini-2.0-flash')

# Utility Functions
def extract_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id):
    try:
        # Try with different proxy configurations if provided in secrets
        proxies = None
        if hasattr(st.secrets, "PROXY_URL"):
            proxies = {
                'http': st.secrets.PROXY_URL,
                'https': st.secrets.PROXY_URL
            }
        
        # First try without proxy
        try:
            return YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            if not proxies:
                raise e
            
            # If first attempt failed and proxy is configured, try with proxy
            return YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
            
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
        return None
    except Exception as e:
        error_msg = str(e)
        if "RequestBlocked" in error_msg:
            st.error("YouTube is blocking our request. This can happen due to rate limiting. Please try again in a few minutes or try a different video.")
        else:
            st.error(f"Error fetching transcript: {error_msg}")
        return None

def generate_summary(text):
    try:
        response = model.generate_content(f"Summarize this transcript concisely: {text[:4000]}")
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def translate_text(text, target_language):
    if not text:
        return "No content available for translation."
    try:
        translator = Translator()
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        st.error(f"Error translating text: {str(e)}")
        return "Translation failed."

def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def generate_pdf(text, summary=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    if summary:
        story.append(Paragraph("Transcript Summary", styles['Heading1']))
        story.append(Paragraph(summary, styles['Normal']))
        story.append(Spacer(1, inch))

    story.append(Paragraph("Full Transcript", styles['Heading1']))
    story.append(Paragraph(text, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_audio(text, language_code):
    try:
        audio_buffer = BytesIO()
        tts = gTTS(text=text, lang=language_code)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def answer_question(context, question):
    try:
        response = model.generate_content(f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely:")
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Unable to answer the question at this time."

def detect_topics(text):
    try:
        response = model.generate_content(f"Extract main topics or keywords: {text[:4000]}")
        return response.text.split(', ')
    except Exception as e:
        st.error(f"Error detecting topics: {str(e)}")
        return []

# Main Application
def main():
    st.set_page_config(page_title="YouTube Transcript Processor with Q&A", layout="wide", initial_sidebar_state="expanded")
    st.title("🎥 Advanced YouTube Video Summarizer with Q&A")

    # Initialize session state
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "qa_answer" not in st.session_state:
        st.session_state.qa_answer = ""
    if "topics" not in st.session_state:
        st.session_state.topics = []

    # Sidebar
    with st.sidebar:
        st.header("🤖 Virtual AI Assistant")
        st.markdown("### Features")
        st.markdown("1. **Extract transcripts**\n2. **Summarize transcripts**\n3. **Translate summaries**\n4. **Generate word clouds**\n5. **Export as PDF/Audio**\n6. **Answer questions**\n7. **Topic detection**")
        target_language = st.selectbox('🌍 Select Translation Language:', ["English", "Telugu", "Tamil", "Malayalam", "Hindi"], index=0)
        language_code_map = {"English": "en", "Telugu": "te", "Tamil": "ta", "Malayalam": "ml", "Hindi": "hi"}
        language_code = language_code_map[target_language]
        enable_audio = st.checkbox("Enable Audio Generation")
        st.info("💡 Tip: Provide a YouTube URL to get started!")

    # Tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📜 Transcript", "📋 Summary", "🌍 Translation", "❓ Q&A"])

    youtube_url = st.text_input("Enter a YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

    with tab1:
        st.header("Welcome to YouTube Transcript Processor!")
        st.write("Analyze video content like never before. Enter a video URL to start.")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            transcript = get_transcript(video_id)
            if transcript:
                st.session_state.transcript_text = ' '.join([entry['text'] for entry in transcript])
                st.session_state.summary = generate_summary(st.session_state.transcript_text)
                st.session_state.topics = detect_topics(st.session_state.transcript_text)
            else:
                st.warning("Transcript could not be retrieved.")
        else:
            st.warning("Invalid YouTube URL.")

    with tab2:
        st.header("📜 Full Transcript")
        st.text_area("Transcript Text", st.session_state.transcript_text, height=200)
        if st.session_state.transcript_text:
            pdf = generate_pdf(st.session_state.transcript_text)
            st.download_button(
                label="📄 Download Full Transcript as PDF",
                data=pdf.getvalue(),
                file_name="full_transcript.pdf",
                mime="application/pdf"
            )

    with tab3:
        st.header("📋 Summary")
        st.write(st.session_state.summary)
        st.write("### Detected Topics:")
        st.write(", ".join(st.session_state.topics))

        if st.session_state.transcript_text and st.session_state.summary:
            pdf = generate_pdf(st.session_state.transcript_text, st.session_state.summary)
            st.download_button(
                label="📄 Download Summary as PDF",
                data=pdf.getvalue(),
                file_name="transcript_summary.pdf",
                mime="application/pdf"
            )

    with tab4:
        translated_summary = translate_text(st.session_state.summary, language_code)
        st.header("🌍 Translated Summary")
        st.write(translated_summary)

        if enable_audio and translated_summary:
            audio_buffer = generate_audio(translated_summary, language_code)
            st.audio(audio_buffer, format="audio/mp3")
            st.download_button(
                label="🎵 Download Translated Summary Audio",
                data=audio_buffer.getvalue(),
                file_name="translated_summary.mp3",
                mime="audio/mp3"
            )

    with tab5:
        st.header("❓ Question & Answer")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if st.session_state.summary:
                st.session_state.qa_answer = answer_question(st.session_state.summary, question)
            else:
                st.warning("Generate a summary first.")
        st.write("**Answer:**", st.session_state.qa_answer)

    # Word Cloud
    if st.session_state.transcript_text:
        st.sidebar.header("🌟 Word Cloud")
        fig = create_word_cloud(st.session_state.transcript_text)
        st.sidebar.pyplot(fig)

if __name__ == "__main__":
    main()
