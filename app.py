import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from supadata import Supadata, SupadataError
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from gtts import gTTS
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta
from collections import Counter
import string

# Load environment variables and configure APIs
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SUPADATA_API_KEY = os.getenv('SUPADATA_API_KEY')

# Validate API keys
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
if not SUPADATA_API_KEY:
    st.error("⚠️ SUPADATA_API_KEY not found in environment variables!")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Supadata client
supadata = Supadata(api_key=SUPADATA_API_KEY) if SUPADATA_API_KEY else None

# Configuration Constants
MAX_CALLS_PER_MINUTE = 10  # Adjusted for better usability
MAX_TRANSCRIPT_LENGTH = 10000  # Characters to process
RATE_LIMIT_WINDOW = timedelta(minutes=1)

# Rate limiting tracker
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = []

def check_rate_limit():
    """Check if we can make an API call"""
    now = datetime.now()
    # Remove calls older than the rate limit window
    st.session_state.api_calls = [
        call_time for call_time in st.session_state.api_calls 
        if now - call_time < RATE_LIMIT_WINDOW
    ]
    
    if len(st.session_state.api_calls) >= MAX_CALLS_PER_MINUTE:
        oldest_call = st.session_state.api_calls[0]
        wait_time = int((oldest_call + RATE_LIMIT_WINDOW - now).total_seconds())
        return False, max(wait_time, 1)
    
    return True, 0

def record_api_call():
    """Record an API call timestamp"""
    st.session_state.api_calls.append(datetime.now())

# Utility Functions
def extract_video_id(url):
    """Extract video ID from YouTube URL with improved pattern matching"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'  # Direct video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript_text(video_id):
    """Get plain text transcript using Supadata API"""
    if not supadata:
        st.error("Supadata client not initialized. Check API key.")
        return None
        
    try:
        transcript = supadata.youtube.transcript(
            video_id=video_id,
            text=True
        )
        
        if hasattr(transcript, 'content'):
            return transcript.content
        else:
            st.warning(f"Processing started with job ID: {transcript.job_id}")
            st.info("This may take a few moments. Please try refreshing in a minute.")
            return None
            
    except SupadataError as e:
        st.error(f"Supadata Error: {e.message}")
        if hasattr(e, 'documentation_url') and e.documentation_url:
            st.info(f"📚 Documentation: {e.documentation_url}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching transcript: {str(e)}")
        return None

def get_video_metadata(video_id):
    """Get video metadata using Supadata API"""
    if not supadata:
        return None
        
    try:
        video = supadata.youtube.video(id=video_id)
        return video
    except SupadataError as e:
        st.error(f"Error fetching video metadata: {e.message}")
        return None
    except Exception as e:
        st.error(f"Metadata error: {str(e)}")
        return None

def truncate_text(text, max_length=MAX_TRANSCRIPT_LENGTH):
    """Intelligently truncate text at sentence boundaries"""
    if len(text) <= max_length:
        return text
    
    # Try to cut at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.8:  # If we can keep 80% of content
        return truncated[:last_period + 1]
    
    return truncated + "..."

def simple_summarize(text, max_sentences=10):
    """Enhanced extractive summarization as fallback"""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    if not sentences:
        return "Transcript too short to summarize."
    
    # Get distributed sentences throughout the content
    if len(sentences) <= max_sentences:
        summary_text = '. '.join(sentences) + '.'
    else:
        indices = [
            int(i * len(sentences) / max_sentences) 
            for i in range(max_sentences)
        ]
        selected_sentences = [sentences[i] for i in indices]
        summary_text = '. '.join(selected_sentences) + '.'
    
    # Extract key topics using word frequency
    topics = simple_topic_detection(text, num_topics=5)
    key_points = "\n\n**Key Topics:** " + ", ".join(topics)
    
    return summary_text + key_points

def generate_summary(text, use_fallback=False):
    """Generate summary using Gemini AI with improved error handling"""
    if not text:
        return "No transcript available to summarize."
        
    if use_fallback:
        st.info("📝 Using simple extractive summary (API-free mode)")
        return simple_summarize(text, max_sentences=10)
    
    can_call, wait_time = check_rate_limit()
    
    if not can_call:
        st.warning(f"⏳ Rate limit reached. Please wait {wait_time} seconds.")
        return simple_summarize(text, max_sentences=10)
    
    try:
        record_api_call()
        
        # Truncate text intelligently
        processed_text = truncate_text(text, 8000)
        
        prompt = f"""Analyze this YouTube video transcript and provide:

**Summary:**
Write a comprehensive 4-6 sentence summary capturing the main message and key insights.

**Key Points:**
- List 5-7 important takeaways from the video
- Each point should be specific and actionable

**Main Topics:**
List 3-5 primary topics or themes discussed

Transcript:
{processed_text}"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        return response.text
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
            st.error("❌ Google Gemini API quota exceeded!")
            st.info("""
            **Solutions:**
            1. ⏰ Wait a few minutes and try again
            2. ✅ Enable 'Use Simple Summary' option
            3. 🔑 Get a new API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
            4. 💳 Upgrade to paid tier for higher limits
            """)
        else:
            st.error(f"⚠️ Error generating summary: {str(e)}")
        
        return simple_summarize(text, max_sentences=10)

def translate_text(text, target_language):
    """Translate text with better error handling"""
    if not text or text == "No content available for translation.":
        return "No content available for translation."
        
    try:
        # Limit text length for translation (deep-translator has a 5000 char limit per request)
        if len(text) > 4500:
            text = text[:4500] + "..."
        
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        st.info("💡 Tip: Translation services have character limits. Try with shorter text.")
        return "Translation failed. Please try again or use a different translation service."

def create_word_cloud(text):
    """Create word cloud with improved styling"""
    if not text:
        return None
        
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white",
            colormap="viridis",
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        return plt
    except Exception as e:
        st.error(f"Word cloud generation failed: {str(e)}")
        return None

def generate_pdf(text, summary=None):
    """Generate PDF with transcript and summary"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("YouTube Transcript Analysis", styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    
    # Timestamp
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.5*inch))

    if summary:
        story.append(Paragraph("Summary", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(summary.replace('\n', '<br/>'), styles['Normal']))
        story.append(Spacer(1, 0.5*inch))

    story.append(Paragraph("Full Transcript", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(text[:15000].replace('\n', '<br/>'), styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_audio(text, language_code):
    """Generate audio from text using gTTS with error handling"""
    try:
        # Limit text length for audio generation
        text_for_audio = text[:5000] if len(text) > 5000 else text
        
        audio_buffer = BytesIO()
        tts = gTTS(text=text_for_audio, lang=language_code, slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        return None

def answer_question(context, question):
    """Answer question based on context using Gemini AI with detailed responses"""
    if not context or not question:
        return "Please provide both context and a question."
        
    can_call, wait_time = check_rate_limit()
    
    if not can_call:
        return f"⏳ Rate limit reached. Please wait {wait_time} seconds before asking another question."
    
    try:
        record_api_call()
        
        # Use more context for better answers
        processed_context = truncate_text(context, 8000)
        
        prompt = f"""You are a helpful assistant analyzing a YouTube video transcript. Answer the user's question in a detailed and comprehensive way.

Context from video transcript:
{processed_context}

User's Question: {question}

Instructions:
1. Provide a thorough, detailed answer based on the context
2. Include specific examples or quotes from the transcript when relevant
3. Structure your answer with clear paragraphs if needed
4. If the question asks for a list, provide numbered or bulleted points
5. Explain concepts clearly as if teaching someone
6. If the context doesn't fully address the question, mention what information is available and what might be missing
7. Use 4-8 sentences or more if needed for completeness

Answer:"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Slightly higher for more detailed responses
                max_output_tokens=1000,  # Increased token limit for detailed answers
                top_p=0.95,
            )
        )
        return response.text.strip()
        
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg:
            return "❌ API quota exceeded. Please wait a few minutes before asking another question."
        return f"Unable to answer: {str(e)}"

def simple_topic_detection(text, num_topics=5):
    """Enhanced keyword extraction as fallback"""
    # Remove punctuation and convert to lowercase
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text_clean.split()
    
    # Expanded stop words list
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
        'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
        'why', 'how', 'so', 'just', 'now', 'very', 'too', 'also', 'than',
        'then', 'there', 'their', 'them', 'from', 'into', 'about', 'more',
        'some', 'like', 'really', 'thing', 'things', 'going', 'know'
    }
    
    # Filter words (minimum length 4)
    filtered_words = [
        word for word in words 
        if word not in stop_words and len(word) > 3
    ]
    
    # Get most common words
    word_counts = Counter(filtered_words)
    topics = [
        word.capitalize() 
        for word, count in word_counts.most_common(num_topics)
    ]
    
    return topics if topics else ["General content"]

def detect_topics(text, use_fallback=False):
    """Detect main topics from text using Gemini AI"""
    if not text:
        return []
        
    if use_fallback:
        return simple_topic_detection(text)
    
    can_call, wait_time = check_rate_limit()
    
    if not can_call:
        st.info("📊 Using simple keyword extraction for topics")
        return simple_topic_detection(text)
    
    try:
        record_api_call()
        
        processed_text = truncate_text(text, 4000)
        
        prompt = f"""Extract 5-7 main topics or key themes from this text. 
        Return them as a comma-separated list of short phrases (2-4 words each).
        Focus on the most important concepts discussed.
        
        Text: {processed_text}"""
        
        response = model.generate_content(prompt)
        topics = [topic.strip() for topic in response.text.split(',')]
        return topics[:7]  # Limit to 7 topics
        
    except Exception as e:
        return simple_topic_detection(text)

# Main Application
def main():
    st.set_page_config(
        page_title="YouTube Transcript Processor", 
        layout="wide", 
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Advanced YouTube Video Analyzer powered by DSNP"
        }
    )
    
    st.title("🎥 Advanced YouTube Video Summarizer with Q&A")
    st.caption("Powered by DSNP")

    # Initialize session state
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "qa_answer" not in st.session_state:
        st.session_state.qa_answer = ""
    if "topics" not in st.session_state:
        st.session_state.topics = []
    if "video_metadata" not in st.session_state:
        st.session_state.video_metadata = None
    if "video_id" not in st.session_state:
        st.session_state.video_id = None

    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Assistant Controls")
        
        # API Status
        api_calls_remaining = max(0, MAX_CALLS_PER_MINUTE - len([
            call_time for call_time in st.session_state.api_calls 
            if datetime.now() - call_time < RATE_LIMIT_WINDOW
        ]))
        
        st.metric(
            "API Calls Available", 
            f"{api_calls_remaining}/{MAX_CALLS_PER_MINUTE}",
            help="Resets every minute"
        )
        
        st.markdown("---")
        
        use_fallback = st.checkbox(
            "🔄 Use Simple Summary (No API)", 
            value=False,
            help="Use basic summarization without consuming AI API quota"
        )
        
        st.markdown("---")
        
        st.markdown("### 🌍 Translation Settings")
        target_language = st.selectbox(
            'Select Language:', 
            ["English", "Spanish", "French", "German", "Telugu", "Tamil", "Malayalam", "Hindi"], 
            index=0
        )
        
        language_code_map = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Telugu": "te", 
            "Tamil": "ta", 
            "Malayalam": "ml", 
            "Hindi": "hi"
        }
        language_code = language_code_map[target_language]
        
        enable_audio = st.checkbox("🔊 Enable Audio Generation")
        
        st.markdown("---")
        
        st.markdown("### ✨ Features")
        st.markdown("""
        - 📜 Extract transcripts
        - 🤖 AI-powered summaries
        - 🌍 Multi-language translation
        - 📄 PDF export
        - 🎵 Audio generation
        - ❓ Intelligent Q&A
        - 📊 Topic detection
        - ☁️ Word cloud visualization
        """)
        
        if api_calls_remaining < 3:
            st.warning("⚠️ Low API quota! Enable 'Simple Summary' mode.")

    # Main Input
    youtube_url = st.text_input(
        "🔗 Enter YouTube Video URL:", 
        placeholder="Paste Your Link Here",
        help="Paste any YouTube video URL or video ID"
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home", 
        "📜 Transcript", 
        "📋 Summary", 
        "🌍 Translation", 
        "❓ Q&A",
        "ℹ️ Video Info"
    ])

    with tab1:
        st.header("Welcome to YouTube Transcript Processor!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            1. Paste a YouTube URL above
            2. Wait for transcript extraction
            3. Explore summaries, translations, and Q&A
            4. Export as PDF or audio
            
            ### 🔥 Powered By
            - DSNP
            
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Pro Tips
            - Enable 'Simple Summary' if you hit API limits
            - Use Q&A to ask specific questions about content
            - Generate audio for language learning
            - Export PDFs for offline reading
            
            ### ⚙️ API Limits
            Free tier limits apply. Monitor your quota in the sidebar.
            """)
        
        if not GOOGLE_API_KEY or not SUPADATA_API_KEY:
            st.error("⚠️ Missing API keys! Please configure your environment variables.")
            st.code("""
# Create a .env file with:
GOOGLE_API_KEY=your_google_api_key
SUPADATA_API_KEY=your_supadata_api_key
            """)

    # Process YouTube URL
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if video_id and video_id != st.session_state.video_id:
            st.session_state.video_id = video_id
            
            with st.spinner("🔄 Fetching transcript and video information..."):
                # Get transcript
                transcript_text = get_transcript_text(video_id)
                
                if transcript_text:
                    st.session_state.transcript_text = transcript_text
                    st.success("✅ Transcript fetched successfully!")
                    
                    # Generate summary
                    with st.spinner("🤖 Generating AI summary..."):
                        st.session_state.summary = generate_summary(
                            st.session_state.transcript_text, 
                            use_fallback=use_fallback
                        )
                    
                    # Detect topics
                    st.session_state.topics = detect_topics(
                        st.session_state.transcript_text,
                        use_fallback=use_fallback
                    )
                else:
                    st.warning("⚠️ Transcript could not be retrieved. The video may not have captions.")
                
                # Get metadata
                video_metadata = get_video_metadata(video_id)
                if video_metadata:
                    st.session_state.video_metadata = video_metadata
        elif not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")

    with tab2:
        st.header("📜 Full Transcript")
        
        if st.session_state.transcript_text:
            st.text_area(
                "Transcript Text", 
                st.session_state.transcript_text, 
                height=300,
                help="Complete transcript extracted from the video"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Words", len(st.session_state.transcript_text.split()))
            with col2:
                st.metric("🔤 Characters", len(st.session_state.transcript_text))
            with col3:
                sentences = len([s for s in st.session_state.transcript_text.split('.') if s.strip()])
                st.metric("📋 Sentences", sentences)
            
            pdf = generate_pdf(st.session_state.transcript_text)
            st.download_button(
                label="📄 Download Full Transcript as PDF",
                data=pdf.getvalue(),
                file_name=f"transcript_{st.session_state.video_id}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("👆 Enter a YouTube URL to view the transcript.")

    with tab3:
        st.header("📋 AI-Generated Summary")
        
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🔄 Regenerate", help="Generate a new summary"):
                    with st.spinner("Generating new summary..."):
                        st.session_state.summary = generate_summary(
                            st.session_state.transcript_text,
                            use_fallback=use_fallback
                        )
                        st.rerun()
            
            st.markdown("---")
            
            if st.session_state.topics:
                st.markdown("### 🏷️ Detected Topics")
                
                # Display topics as tags
                topics_html = " ".join([
                    f'<span style="background-color:#e1f5ff;color:#000000;padding:5px 10px;margin:3px;border-radius:15px;display:inline-block;">{topic}</span>'
                    for topic in st.session_state.topics
                ])
                st.markdown(topics_html, unsafe_allow_html=True)
            
            st.markdown("---")

            if st.session_state.transcript_text:
                pdf = generate_pdf(
                    st.session_state.transcript_text, 
                    st.session_state.summary
                )
                st.download_button(
                    label="📄 Download Summary + Transcript as PDF",
                    data=pdf.getvalue(),
                    file_name=f"summary_{st.session_state.video_id}.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("👆 Generate a transcript first to see the AI summary.")

    with tab4:
        st.header("🌍 Translated Summary")
        
        if st.session_state.summary:
            st.info(f"Translating to: **{target_language}**")
            
            with st.spinner("Translating..."):
                translated_summary = translate_text(
                    st.session_state.summary, 
                    language_code
                )
            
            st.markdown("### Translated Content")
            st.write(translated_summary)

            if enable_audio and translated_summary and "failed" not in translated_summary.lower():
                st.markdown("---")
                st.markdown("### 🎵 Audio Version")
                
                with st.spinner("Generating audio..."):
                    audio_buffer = generate_audio(translated_summary, language_code)
                
                if audio_buffer:
                    st.audio(audio_buffer, format="audio/mp3")
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=audio_buffer.getvalue(),
                        file_name=f"summary_{language_code}.mp3",
                        mime="audio/mp3"
                    )
        else:
            st.info("👆 Generate a summary first to see the translation.")

    with tab5:
        st.header("❓ Ask Questions About the Video")
        
        if st.session_state.summary or st.session_state.transcript_text:
            st.markdown("💡 **Ask detailed questions about the video content and get comprehensive answers!**")
            
            # Example questions
            with st.expander("📌 Example Questions You Can Ask"):
                st.markdown("""
                - What are the main points discussed in this video?
                - Can you explain [specific topic] mentioned in the video?
                - What examples or case studies were provided?
                - What are the key takeaways from this video?
                - How does the speaker explain [concept]?
                - What recommendations or advice does the video give?
                - What problems does the video address and what solutions does it propose?
                """)
            
            question = st.text_input(
                "Your Question:",
                placeholder="e.g., What are the main points discussed and how are they explained?"
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                ask_button = st.button("🔍 Get Detailed Answer", type="primary", use_container_width=True)
            
            with col2:
                use_full_transcript = st.checkbox(
                    "Use full transcript", 
                    value=False,
                    help="Use complete transcript instead of summary for more detailed context"
                )
            
            if ask_button:
                if question:
                    # Choose context based on user preference
                    if use_full_transcript and st.session_state.transcript_text:
                        context = st.session_state.transcript_text
                        st.info("🔍 Analyzing full transcript for detailed answer...")
                    else:
                        context = st.session_state.summary or st.session_state.transcript_text
                        st.info("🔍 Analyzing summary for answer...")
                    
                    with st.spinner("🤔 Generating detailed answer..."):
                        st.session_state.qa_answer = answer_question(context, question)
                else:
                    st.warning("Please enter a question first.")
            
            if st.session_state.qa_answer:
                st.markdown("---")
                st.markdown("### 💬 Detailed Answer")
                
                # Display answer in a nice container
                st.markdown(f"""
                <div style="background-color: #f0f7ff; color: #000000; padding: 20px; border-radius: 10px; border-left: 5px solid #1976d2;">
                    {st.session_state.qa_answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Additional options
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("🗑️ Clear Answer"):
                        st.session_state.qa_answer = ""
                        st.rerun()
                
                with col2:
                    if st.button("📋 Copy Answer"):
                        st.code(st.session_state.qa_answer, language=None)
                
                # Option to translate answer
                st.markdown("---")
                st.markdown("#### 🌍 Translate Answer")
                translate_answer = st.checkbox("Translate this answer")
                
                if translate_answer:
                    answer_lang = st.selectbox(
                        "Select language for answer:",
                        ["Spanish", "French", "German", "Telugu", "Tamil", "Malayalam", "Hindi"],
                        key="answer_lang"
                    )
                    
                    answer_lang_codes = {
                        "Spanish": "es", "French": "fr", "German": "de",
                        "Telugu": "te", "Tamil": "ta", "Malayalam": "ml", "Hindi": "hi"
                    }
                    
                    if st.button("Translate Answer"):
                        with st.spinner("Translating..."):
                            translated_answer = translate_text(
                                st.session_state.qa_answer,
                                answer_lang_codes[answer_lang]
                            )
                            st.success(f"**Translated to {answer_lang}:**")
                            st.write(translated_answer)
        else:
            st.info("👆 Generate a transcript first to enable Q&A.")
            
            # Preview of feature
            st.markdown("""
            ### 🎯 What you can do with Q&A:
            
            - **Ask specific questions** about video content
            - **Get detailed explanations** of concepts discussed
            - **Extract key information** without watching the full video
            - **Clarify doubts** about topics mentioned
            - **Get comprehensive answers** with examples and context
            - **Translate answers** to your preferred language
            
            Simply enter a YouTube URL above to get started!
            """)

    with tab6:
        st.header("ℹ️ Video Information")
        
        if st.session_state.video_metadata:
            metadata = st.session_state.video_metadata
            
            # Video Title
            st.subheader(f"📹 {getattr(metadata, 'title', 'Unknown Title')}")
            
            # Metadata in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("👤 Channel", getattr(metadata, 'channel_name', 'N/A'))
                st.metric("⏱️ Duration", getattr(metadata, 'duration', 'N/A'))
            
            with col2:
                views = getattr(metadata, 'views', 'N/A')
                if isinstance(views, int):
                    views = f"{views:,}"
                st.metric("👁️ Views", views)
                
                st.metric("📅 Published", getattr(metadata, 'published_at', 'N/A'))
            
            with col3:
                st.metric("🆔 Video ID", st.session_state.video_id)
            
            # Video Description
            if hasattr(metadata, 'description') and metadata.description:
                with st.expander("📝 View Full Description"):
                    st.write(metadata.description)
            
            # Thumbnail
            if hasattr(metadata, 'thumbnail_url'):
                st.image(
                    metadata.thumbnail_url, 
                    caption="Video Thumbnail",
                    use_container_width=True
                )
            
            # Additional metadata if available
            st.markdown("---")
            st.markdown("### 📊 Additional Details")
            
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                if hasattr(metadata, 'category'):
                    st.write(f"**Category:** {metadata.category}")
                if hasattr(metadata, 'language'):
                    st.write(f"**Language:** {metadata.language}")
            
            with details_col2:
                if hasattr(metadata, 'tags'):
                    st.write(f"**Tags:** {', '.join(metadata.tags[:5])}")
                if hasattr(metadata, 'license'):
                    st.write(f"**License:** {metadata.license}")
        else:
            st.info("👆 Enter a YouTube URL to see video information.")
            
            # Show placeholder
            st.markdown("""
            ### What you'll see here:
            - 📹 Video title and channel
            - 👁️ View count and publish date
            - ⏱️ Duration
            - 📝 Description
            - 🖼️ Thumbnail
            - 📊 Additional metadata
            """)

    # Word Cloud in Sidebar (bottom)
    if st.session_state.transcript_text:
        st.sidebar.markdown("---")
        st.sidebar.header("☁️ Word Cloud")
        
        with st.sidebar:
            fig = create_word_cloud(st.session_state.transcript_text)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Word cloud generation failed")

if __name__ == "__main__":
    main()