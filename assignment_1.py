import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import logging
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import io
from typing import List
from more_itertools import chunked  
from io import BytesIO
from streamlit_tags import st_tags

# -----------------------------------
# Configuration and Setup
# -----------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AI Sentiment Analysis Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# -----------------------------------
# Sentiment Analyzer Class
# -----------------------------------

class SentimentAnalyzer:
    """A professional sentiment analysis tool using Google's Gemini AI."""
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def analyze_sentiment_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Analyze sentiment of a batch of texts using single-prompt batch approach.

        Args:
            texts (List[str]): List of texts to analyze.
            batch_size (int): Number of texts per API call.

        Returns:
            List[Dict]: Sentiment results per input text.
        """
        all_results = []

        for chunk in [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]:
            numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(chunk)])

            prompt = f"""
            You are a professional sentiment analysis system.

            Analyze the sentiment for each of the numbered texts below. For each, return the sentiment as one of `"positive"`, `"negative"`, or `"neutral"` and a confidence score between 0 and 1.

            Your response must be a **JSON array**, where each item corresponds to the respective numbered input, like:

            [
            {{"sentiment": "positive", "confidence": 0.95}},
            ...
            ]

            Texts to analyze:
            {numbered_texts}
            """

            try:
                response = self.model.generate_content(prompt)
                content = response.text.strip()

                # Clean JSON block if wrapped
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]

                parsed = json.loads(content)
                if isinstance(parsed, list):
                    all_results.extend(parsed)
                else:
                    logger.warning("Response is not a list")
                    all_results.extend([{"sentiment": "error", "confidence": 0.0}] * len(chunk))

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                all_results.extend([{"sentiment": "error", "confidence": 0.0}] * len(chunk))
            except Exception as e:
                logger.error(f"API error: {type(e).__name__}: {e}")
                all_results.extend([{"sentiment": f"Error: {e}", "confidence": 0.0}] * len(chunk))

        return all_results

    def suggest_tags_from_batch(self, texts: List[str], batch_size: int = 10) -> List[str]:
        accumulated_tags = []  # tag tổng hợp qua các batch

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            sample_reviews = "\n".join([f"- {t}" for t in batch])

            tag_context = f"\nExisting tags so far: {accumulated_tags}" if accumulated_tags else ""

            prompt = f"""
            You are a customer feedback analyst.

            Based on the following reviews, suggest a concise list of general feedback tags that categorize the content.
            Each tag should be broad (e.g., "product defects", "pricing", "usability", "customer service").

            Reviews:
            {sample_reviews}

            {tag_context}

            Only suggest **new** tags that are not already in the existing tag list.
            If no new tags are needed, return an empty list.

            Return a JSON array like:
            ["new tag 1", "new tag 2"]
            """

            try:
                response = self.model.generate_content(prompt)
                content = response.text.strip()

                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()

                new_tags = json.loads(content)

                if isinstance(new_tags, list):
                    for tag in new_tags:
                        normalized_tag = tag.strip().lower()
                        if normalized_tag not in [t.lower() for t in accumulated_tags]:
                            accumulated_tags.append(tag.strip())
            except Exception as e:
                logger.error(f"Failed to suggest tags in batch {i // batch_size + 1}: {e}")

        return accumulated_tags

    def assign_tags_to_texts(self, texts: List[str], tag_list: List[str], batch_size: int = 1) -> List[Dict]:
        """
        Assign tags from a given tag list to each review, with optional batch size.

        Args:
            texts (List[str]): List of reviews.
            tag_list (List[str]): Tags to consider when labeling.
            batch_size (int): Process batch size (default is 1 for per-review tagging).

        Returns:
            List[Dict]: List of {"tags": [...]}
        """
        import json
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                prompt = f"""
                You are a professional tagger for customer feedback.

                Given the following list of tags: {tag_list}

                Assign the most relevant tags from the list to the review below. Return 1–3 tags max.

                Review: "{text}"

                Respond only in JSON format:
                {{
                    "tags": ["tag1", "tag2"]
                }}
                """

                try:
                    response = self.model.generate_content(prompt)
                    content = response.text.strip()

                    if content.startswith("```json"):
                        content = content[7:-3].strip()
                    elif content.startswith("```"):
                        content = content[3:-3].strip()

                    parsed = json.loads(content)
                    tags = parsed.get("tags", [])
                    if not isinstance(tags, list):
                        tags = ["error"]
                except Exception as e:
                    logger.warning(f"Error tagging review: {text[:30]}... | {e}")
                    tags = ["error"]

                results.append({"tags": tags})

        return results


# -----------------------------------
# UI Styling
# -----------------------------------

def apply_custom_css():
    """Apply custom CSS styling for a modern, professional look."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            background: white;
            font-family: 'Inter', sans-serif;
        }
        
        .stTabs [role="tab"] {
            color: black !important;
            font-weight: 600;
        }

        .main .block-container {
            padding: 2rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            margin: 1rem;
        }

        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        .subtitle {
            text-align: center;
            color: #6c757d;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }

        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin: 0.5rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }

        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }

        .css-1cpxqw2 {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .css-1cpxqw2:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        }

        .stSuccess, .stError, .stInfo {
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        .stSuccess {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #28a745;
        }

        .stError {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 1px solid #dc3545;
        }

        .stInfo {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border: 1px solid #17a2b8;
        }

        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .rotating { animation: rotate 2s linear infinite; }
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: .7; }
        }

        .welcome-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: 2rem 0;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }

        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------
# UI Components
# -----------------------------------

def create_animated_header():
    """Create an animated header with modern styling."""
    st.markdown("""
        <div class="main-header">🧠 AI Sentiment Analysis Studio</div>
        <div class="subtitle">
            ✨ Powered by Google Gemini AI • Beautiful • Professional • Fast
        </div>
    """, unsafe_allow_html=True)

def create_welcome_section():
    """Display a welcome section with feature highlights."""
    st.markdown("""
        <h2 style="color: #667eea; margin-bottom: 1rem;">🚀 Welcome to AI Sentiment Analysis Studio</h2>
        <p style="font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem;">
            Transform your customer feedback into actionable insights with our advanced AI-powered sentiment analysis platform.
        </p>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    features = [
        ("⚡", "Lightning Fast", "Process thousands of reviews in seconds with optimized batch processing."),
        ("🎯", "Highly Accurate", "Advanced AI models provide 95%+ accuracy with confidence scoring."),
        ("📊", "Beautiful Analytics", "Interactive charts and visualizations for deep insights."),
        ("🛡️", "Secure & Private", "Your data is processed securely with enterprise-grade security.")
    ]
    
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(f"""
            <div class="metric-card"; style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem;">{icon}</div>
                <h4 style="color: #495057; margin: 0.5rem 0;">{title}</h4>
                <p style="color: #6c757d;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

def create_sentiment_beautiful_metrics(results_df: pd.DataFrame, original_df: pd.DataFrame):
    """Display animated metric cards for analysis summary."""
    df = original_df.copy() # Work on a copy
    cols = st.columns(4)
    
    total_entries = len(results_df)
    counts = {
        'positive': len(results_df[results_df['sentiment'] == 'positive']),
        'neutral': len(results_df[results_df['sentiment'] == 'neutral']),
        'negative': len(results_df[results_df['sentiment'] == 'negative'])
    }
    
    # Ensure 'platform' column exists in df (the copy) for this function's scope
    if 'platform' not in df.columns:
        df['platform'] = 'Unknown' # Modify the copy

    avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
    platform_count = len(df['platform'].unique()) if 'platform' in df.columns else 1
    
    metrics = [
        ("📊", total_entries, "Total Reviews", "#667eea"),
        ("😊", counts['positive'], f"Positive ({counts['positive'] / (counts['negative'] + counts['neutral'] + counts['positive']) * 100:.1f}%)", "#28a745"),
        ("🌐", platform_count, "Platforms", "#17a2b8"),
        ("⭐", f"{avg_rating:.1f}", "Avg. Rating", "#ffc107")
    ]
    
    for col, (icon, value, label, color) in zip(cols, metrics):
        col.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color};">{value}</div>
                    <div style="color: #6c757d; font-weight: 500;">{label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def create_tag_beautiful_metrics(results_df: pd.DataFrame):
    """Display animated metric cards for tag classification summary."""
    import collections

    cols = st.columns(4)
    total_reviews = len(results_df)

    # Flatten toàn bộ tags
    all_tags = [tag for tags in results_df["tags"] if isinstance(tags, list) for tag in tags]
    tag_counts = collections.Counter(all_tags)

    total_tags = sum(tag_counts.values())
    avg_tags_per_review = total_tags / total_reviews if total_reviews > 0 else 0
    most_common_tag, most_common_count = tag_counts.most_common(1)[0] if tag_counts else ("N/A", 0)

    metrics = [
        ("📝", total_reviews, "Total Reviews", "#667eea"),
        ("🏷️", total_tags, "Total Tags Assigned", "#6f42c1"),
        ("⚖️", f"{avg_tags_per_review:.2f}", "Avg. Tags per Review", "#17a2b8"),
        ("🔥", f"{most_common_count}", f"{most_common_tag}", "#dc3545")
    ]

    for col, (icon, value, label, color) in zip(cols, metrics):
        col.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color};">{value}</div>
                    <div style="color: #6c757d; font-weight: 500;">{label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def create_beautiful_sidebar():
    """Create a modern sidebar for configuration."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #667eea; margin-bottom: 0.5rem;">⚙️ Configuration</h2>
                <p style="color: #6c757d; font-size: 0.9rem;">Configure your analysis settings</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📂 Page Selection")
        selected_page = st.sidebar.selectbox("### Choose One Page", ["🏠 Home", "📊 Sentiment Analysis", "🛠️ Tag Classification"])
        
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Google Gemini API Key", type="password",
            help="🔒 Your API key is secure and never stored", placeholder="Enter your Gemini API key..."
        )
        
        st.markdown("### 🤖 Model Selection")
        model_options = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
        selected_model = st.selectbox(
            "Select Gemini Model",
            model_options,
            index=0,
            help="Choose the Gemini model you want to use based on your needs"
        )
        
        st.markdown("### 📤 File Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File", type=["csv"], help="📊 Upload a CSV file with a 'review' column"
        )
                
        st.markdown("### 🎛️ Processing Options")
        batch_size = st.slider("Batch Size", 1, 50, 10, help="⚡ Number of reviews to process simultaneously")
        show_confidence = st.checkbox("Show Confidence Scores", value=True, help="📊 Display AI confidence levels")
        show_advanced_charts = st.checkbox("Advanced Visualizations", value=True, help="📈 Show additional charts and analytics")
        
        return selected_page, api_key, selected_model, uploaded_file, batch_size, show_confidence, show_advanced_charts

# -----------------------------------
# Visualization Functions
# -----------------------------------

def create_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    # Validate input DataFrame
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    required_columns = ['platform', 'sentiment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Calculate sentiment counts and percentages
    sentiment_counts = (df.groupby(['platform', 'sentiment'])
                       .size()
                       .reset_index(name='count'))
    
    # Calculate total counts per platform
    total_counts = sentiment_counts.groupby('platform')['count'].sum().reset_index(name='total')
    
    # Merge to calculate percentages
    sentiment_counts = sentiment_counts.merge(total_counts, on='platform')
    sentiment_counts['percentage'] = (sentiment_counts['count'] / sentiment_counts['total'] * 100).round(2)

    # Initialize figure
    fig = go.Figure()

    # Define color mapping
    colors: Dict[str, str] = {
        'positive': '#28a745',
        'neutral': '#ffc107',
        'negative': '#dc3545'
    }

    # Add bar traces for each sentiment
    for sentiment in sentiment_counts['sentiment'].unique():
        sent_data = sentiment_counts[sentiment_counts['sentiment'] == sentiment]
        fig.add_trace(go.Bar(
            x=sent_data['platform'],
            y=sent_data['percentage'],
            name=sentiment.capitalize(),
            marker_color=colors.get(sentiment, '#808080'),
            text=sent_data['percentage'].apply(lambda x: f'{x}%'),
            textposition='auto',
            hovertemplate=(
                f'<b>%{{x}}</b><br>'
                f'Sentiment: {sentiment.capitalize()}<br>'
                f'Percentage: %{{y}}%<br>'
                f'Count: %{{customdata}}<extra></extra>'
            ),
            customdata=sent_data['count']
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="🎯 Sentiment Distribution by Platform",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis=dict(
            title='Platform',
            title_font=dict(color='#2c3e50'), 
            tickfont=dict(color='#2c3e50'),
            tickangle=45
        ),
        yaxis=dict(
            title='Percentage (%)',
            title_font=dict(color='#2c3e50'), 
            tickfont=dict(color='#2c3e50'),
            range=[0, 100],
            gridcolor='rgba(200,200,200,0.3)'
        ),
        barmode='group',
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(b=150)  # Add margin for rotated x-axis labels
    )

    return fig

def create_confidence_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create a histogram for confidence score distribution."""
    if 'confidence' not in df.columns:
        return None
    
    fig = px.histogram(df, x='confidence', nbins=20, title="🎯 Confidence Score Distribution", color_discrete_sequence=['#667eea'])
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        title={'font': {'size': 20, 'color': '#2c3e50'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(title="Confidence Score", title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50')),
        yaxis=dict(title="Frequency", title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400
    )
    return fig

def create_rating_chart(original_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart for rating distribution by platform."""
    df = original_df.copy()
    if 'platform' not in df.columns:
        df['platform'] = 'Unknown'  # Fallback platform name if column is missing
    
    rating_counts = df.groupby(['platform', 'rating']).size().reset_index(name='count')
    
    fig = px.bar(
        rating_counts, x='platform', y='count', color='rating', title="⭐ Rating Distribution by Platform",
        color_continuous_scale=px.colors.sequential.Viridis,
        text='count'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside', textfont=dict(color='#34495e'))
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        title={'font': {'size': 20, 'color': '#2c3e50'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(title="Platform", title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50')),
        yaxis=dict(title="Number of Reviews", title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500
    )
    return fig

def create_tag_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart for tag distribution from classified review results."""
    from collections import Counter
    import plotly.express as px
    import plotly.graph_objects as go

    # Flatten list of tags
    all_tags = [tag for tags in results_df["tags"] if isinstance(tags, list) for tag in tags]
    tag_counts = Counter(all_tags)

    # Convert to DataFrame
    tag_df = pd.DataFrame(tag_counts.items(), columns=['tag', 'count'])
    tag_df = tag_df.sort_values('count', ascending=False).head(15)  # Top 15 tags

    fig = px.bar(
        tag_df, x='tag', y='count', title="🏷️ Tag Distribution",
        color='count', color_continuous_scale=px.colors.sequential.Viridis,
        text='count'
    )

    fig.update_traces(
        texttemplate='%{text}', textposition='outside',
        textfont=dict(color='#34495e')
    )

    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        title={'font': {'size': 20, 'color': '#2c3e50'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(title="Tag", title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50')),
        yaxis=dict(title="Number of Occurrences", title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )

    return fig

# -----------------------------------
# Main Application
# -----------------------------------

def home_page():
    create_welcome_section()
    st.markdown("""
    <div div class="metric-card"; style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #0c5460;">📋 How to Get Started</h3>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    steps = [
        ("1️⃣ Prepare Your Data", "Upload a CSV file with a column named 'review' containing your text data."),
        ("2️⃣ Configure API", "Enter your Google Gemini API key in the sidebar configuration panel."),
        ("3️⃣ Analyze & Export", "Click analyze to process your data and download beautiful reports.")
    ]
    
    for col, (title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.7); padding: 1.5rem; border-radius: 10px;">
            <h4 style="color: #0c5460; margin-top: 0;">{title}</h4>
            <p style="color: #495057;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div div class="metric-card"; style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #0c5460;">📊 Sample Data Format</h3>
    </div>
    """, unsafe_allow_html=True)
    
    sample_data = pd.DataFrame({
        'review': [
            '✨ Amazing product! Excellent quality and fast delivery.',
            '🤔 The service was okay, nothing particularly special.',
            '😞 Disappointing experience, would not recommend to others.'
        ],
        'platform': ['Google Review', 'Facebook', 'Glassdoor'],
        'rating': [5, 3, 1],
        'date': ['2025-06-01', '2025-06-02', '2025-06-03']
    })
    gb = GridOptionsBuilder.from_dataframe(sample_data)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    grid_options = gb.build()
    AgGrid(sample_data, gridOptions=grid_options, theme="alpine", height=300, allow_unsafe_jscode=True)

def sentiment_page(api_key, selected_model, uploaded_file, batch_size, show_confidence, show_advanced_charts):
    # Initialize session state for processed DataFrame
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'uploaded_file_hash' not in st.session_state:
        st.session_state.uploaded_file_hash = None

    if not api_key:
        st.markdown("""
            <div class="metric-card"; style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 2rem 0;">
                <h4 style="color: #856404; margin: 0;">🔐 API Key Required</h4>
                <p style="color: #856404; margin: 0.5rem 0 0 0;">
                    Please enter your Google Gemini API key in the sidebar to continue.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    if uploaded_file:
        current_file_hash = uploaded_file.name + str(uploaded_file.size)

    # Check if a new file has been uploaded or if no file is processed yet
    if uploaded_file and current_file_hash != st.session_state.uploaded_file_hash:
        st.session_state.processed_df = None # Clear previous data if new file
        st.session_state.uploaded_file_hash = current_file_hash

    if st.session_state.processed_df is None: # If no processed_df yet
        if uploaded_file is None:
            st.markdown("""
                <div class="metric-card"; style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                            padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 2rem 0;">
                    <h4 style="color: #856404; margin: 0;">📂 No File Uploaded</h4>
                    <p style="color: #856404; margin: 0.5rem 0 0 0;">
                        Please upload a file to proceed. Use the file uploader above to get started.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            return

        # Process the newly uploaded file
        try:
            original_df = pd.read_csv(uploaded_file)
            original_df.columns = original_df.columns.str.strip().str.lower() # Normalize column names
            if 'platform' not in original_df.columns:
                original_df['platform'] = 'Unknown'
                st.warning("⚠️ 'platform' column missing. Using 'Unknown' as default.")
            if "review" not in original_df.columns:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
                        <h4 style="color: #721c24; margin: 0;">❌ Column Error</h4>
                        <p style="color: #721c24; margin: 0.5rem 0 0 0;">
                            The uploaded file must contain a column named 'review'.<br>
                            <strong>Available columns:</strong> {", ".join(original_df.columns)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.stop()
            st.session_state.processed_df = original_df # Store for future reruns
            # original_df is now set for this run too
        except Exception as e:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 2rem; border-radius: 15px; border-left: 4px solid #dc3545;">
                    <h4 style="color: #721c24; margin-bottom: 0;">💥 Processing Error</h4>
                    <p style="color: #721c24; margin: 0.5rem 0 0 0;">An error occurred while processing your file: <code>{e}</code></p>
                    <p style="color: #721c24; margin: 0.5rem;0 0; font-size: 0.9rem;">Please check your file format and try again.</p>
                </div>
            """, unsafe_allow_html=True)
            logger.error(f"File processing error: {e}")
            st.session_state.processed_df = None # Clear on error
            return # Exit on error
    
    original_df = st.session_state.processed_df # Ensure original_df always points to the session state
    if original_df is None: # Exit if no DataFrame is available
        return

    try:
        st.markdown("""
            <div class="metric-card"; style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #0c5460;">📋 Data Preview</h3>
            </div>
        """, unsafe_allow_html=True)
        print(f"DEBUG: original_df columns at Data Preview: {original_df.columns.tolist()}")
        
        cols = st.columns(3)
        metrics = [
            ("Total Reviews", len(original_df), "#667eea"),
            ("Valid Reviews", original_df['review'].notna().sum(), "#28a745"),
            ("Avg. Rating", f"{original_df['rating'].mean():.1f}" if 'rating' in original_df.columns else "N/A", "#ffc107")
        ]
        
        for col, (label, value, color) in zip(cols, metrics):
            col.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: {color}; font-weight: 700;">{value}</div>
                        <div style="color: #6c757d;">{label}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        gb = GridOptionsBuilder.from_dataframe(original_df.head(10))
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
        grid_options = gb.build()
        AgGrid(original_df.head(10), gridOptions=grid_options, theme="alpine", height=300, allow_unsafe_jscode=True)
        
        _, col, _ = st.columns([1, 4, 1])
        with col:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                try:
                    analyzer = SentimentAnalyzer(selected_model, api_key)
                except Exception as e:
                    st.markdown(f"""
                        <div class="metric-card"; style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
                            <h4 style="color: #721c24; margin: 0;">⚠️ Initialization Error</h4>
                            <p style="color: #721c24; margin: 0.5rem 0 0 0;">Failed to initialize analyzer: {e}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    return
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">🤖 AI Processing in Progress</h3>
                        <p style="color: #6c757d;">Our advanced AI is analyzing your reviews...</p>
                    </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_container = st.empty()
                valid_reviews = []
                meta_info = []
                results = []
                total_reviews = len(original_df)
                start_time = time.time()
                
                for i, row in original_df.iterrows():
                    progress = (i + 1) / total_reviews
                    progress_bar.progress(progress)
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time / progress if progress > 0 else 0
                    remaining_time = max(0, estimated_total - elapsed_time)
                    
                    status_container.markdown(f"""
                        <div class="metric-card"; style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 1rem 0;">
                            <div style="font-size: 1.2rem; color: #667eea; font-weight: 600;">Processing Review {i + 1} of {total_reviews}</div>
                            <div style="color: #6c757d; margin-top: 0.5rem;">⏱️ Estimated time remaining: {remaining_time:.0f}s</div>
                            <div style="width: 100%; background: #e9ecef; border-radius: 10px; margin-top: 1rem;">
                                <div style="width: {progress*100:.1f}%; height: 8px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; transition: width 0.3s ease;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    review = row['review']
                    if pd.isna(review) or review.strip() == '':
                        results.append({
                            "review": review,
                            "platform": row.get('platform', 'Unknown'),
                            "rating": row.get('rating', 0),
                            "sentiment": "neutral",
                            "confidence": 0.0
                        })
                        continue
                    
                    valid_reviews.append(review)
                    meta_info.append({
                        "platform": row.get('platform', 'Unknown'),
                        "rating": row.get('rating', 0),
                        "review": review
                    })
                    
                batch_results = analyzer.analyze_sentiment_batch(valid_reviews, batch_size)
                    
                for meta, analysis in zip(meta_info, batch_results):
                    result_entry = {
                        "review": meta["review"],
                        "platform": meta["platform"],
                        "rating": meta["rating"],
                        "sentiment": analysis.get("sentiment", "neutral")
                    }
                    if show_confidence:
                        result_entry["confidence"] = analysis.get("confidence", 0.0)
                    results.append(result_entry)
                
                progress_bar.empty()
                status_container.empty()
                
                processing_time = time.time() - start_time
                st.markdown(f"""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);">
                        <h2 style="color: #155724; margin-bottom: 1rem;">🎉 Analysis Complete!</h2>
                        <p style="color: #155724; font-size: 1.1rem;">
                            Successfully processed <strong>{total_reviews}</strong> reviews in <strong>{processing_time:.1f}</strong> seconds
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                results_df = pd.DataFrame(results)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Analysis Summary</h3>
                        <p style="color: #6c757d;">Key insights from your data</p>
                    </div>
                """, unsafe_allow_html=True)
                create_sentiment_beautiful_metrics(results_df, original_df)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📋 Detailed Results</h3>
                        <p style="color: #6c757d;">Detailed sentiment analysis results</p>
                    </div>
                """, unsafe_allow_html=True)
                
                cell_style_jscode = JsCode("""
                    function(params) {
                        if (params.value === 'positive') {
                            return { 'color': 'white', 'backgroundColor': '#28a745', 'fontWeight': 'bold' };
                        } else if (params.value === 'negative') {
                            return { 'color': 'white', 'backgroundColor': '#dc3545', 'fontWeight': 'bold' };
                        } else if (params.value === 'neutral') {
                            return { 'color': 'black', 'backgroundColor': '#ffc107', 'fontWeight': 'bold' };
                        }
                        return null;
                    }
                """)
                
                gb = GridOptionsBuilder.from_dataframe(results_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
                gb.configure_column("sentiment", headerName="Sentiment", cellStyle=cell_style_jscode)
                grid_options = gb.build()
                
                AgGrid(results_df, gridOptions=grid_options, theme="alpine", height=400, fit_columns_on_grid_load=True, allow_unsafe_jscode=True)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Interactive Analytics Dashboard</h3>
                        <p style="color: #6c757d;">Visualize your sentiment analysis results</p>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(2)
                with cols[0]:
                    sentiment_fig = create_sentiment_chart(results_df)
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                
                with cols[1]:
                    rating_fig = create_rating_chart(original_df)
                    st.plotly_chart(rating_fig, use_container_width=True)
                
                if show_advanced_charts and show_confidence and 'confidence' in results_df.columns:
                    st.markdown("""
                        <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                            <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Advanced Analytics</h3>
                            <p style="color: #6c757d;">Additional insights and visualizations</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        confidence_fig = create_confidence_chart(results_df)
                        if confidence_fig:
                            st.plotly_chart(confidence_fig, use_container_width=True)
                    
                    with cols[1]:
                        fig_scatter = px.scatter(
                            results_df, x='confidence', y='sentiment', color='sentiment',
                            title="🎯 Sentiment vs Confidence Analysis",
                            color_discrete_map={'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
                        )
                        fig_scatter.update_layout(
                            font=dict(family="Inter, sans-serif", size=12),
                            title={'font': {'size': 20, 'color': '#2c3e50'}, 'x': 0.5, 'xanchor': 'center'},
                            xaxis={'title': {'text': "Confidence", 'font': {'color': '#2c3e50'}}, 'tickfont': {'color': '#2c3e50'}},
                            yaxis={'title': {'text': "Sentiment", 'font': {'color': '#2c3e50'}}, 'tickfont': {'color': '#2c3e50'}},
                            legend={'title': {'text': "Sentiment", 'font': {'color': '#2c3e50'}}, 'font': {'color': '#2c3e50'}}, showlegend=True,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📥 Export Your Results</h3>
                        <p style="color: #6c757d;">Download your analysis in multiple formats</p>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(3)
                with cols[0]:
                    csv_output = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download CSV Report", data=csv_output,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", use_container_width=True
                    )
                
                with cols[1]:
                    summary_stats = {
                        'analysis_summary': {
                            'total_reviews': total_reviews,
                            'processing_time_seconds': round(processing_time, 2),
                            'analysis_date': datetime.now().isoformat()
                        },
                        'sentiment_distribution': results_df['sentiment'].value_counts().to_dict(),
                        'confidence_stats': {
                            'average_confidence': float(results_df['confidence'].mean()) if 'confidence' in results_df.columns else None,
                            'high_confidence_count': int((results_df['confidence'] >= 0.8).sum()) if 'confidence' in results_df.columns else None
                        }
                    }
                    json_output = json.dumps(summary_stats, indent=2, ensure_ascii=False).encode('utf-8')
                    st.download_button(
                        label="📋 Download JSON Summary", data=json_output,
                        file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json", use_container_width=True
                    )
                
                with cols[2]:
                    try:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
                            summary_df = pd.DataFrame([
                                ['Total Reviews', total_reviews],
                                ['Processing Time (s)', round(processing_time, 2)],
                                ['Positive Reviews', len(results_df[results_df['sentiment'] == 'positive'])],
                                ['Neutral Reviews', len(results_df[results_df['sentiment'] == 'neutral'])],
                                ['Negative Reviews', len(results_df[results_df['sentiment'] == 'negative'])]
                            ], columns=['Metric', 'Value'])
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        st.download_button(
                            label="📈 Download Excel Report", data=excel_buffer.getvalue(),
                            file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True
                        )
                    except ImportError:
                        st.markdown("""
                            <div class="metric-card"; style="background: linear-gradient(135deg, #d1ecf1 0%; #bee5eb 100%); padding: 1rem; border-radius: 10px; border-left: 1px solid #17a2b8;">
                                <p style="color: #0c5460;">📋 Install xlsxwriter for Excel export: <code>pip install xlsxwriter</code></p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0; box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);">
                        <h3 style="color: white; margin-bottom: 1rem;">🎊 Thank You for Using AI Sentiment Analysis Studio!</h3>
                        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem;">
                            Your analysis is complete. Feel free to analyze more data or download your results.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
            <div class="metric-card"; style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 2rem; border-radius: 15px; border-left: 4px solid #dc3545;">
                <h4 style="color: #721c24; margin-bottom: 0;">💥 Processing Error</h4>
                <p style="color: #721c24; margin: 0.5rem 0 0 0;">An error occurred while processing your file: <code>{e}</code></p>
                <p style="color: #721c24; margin: 0.5rem;0 0; font-size: 0.9rem;">Please check your file format and try again.</p>
            </div>
        """, unsafe_allow_html=True)
        logger.error(f"File processing error: {e}")

def classify_page(api_key, selected_model, uploaded_file, batch_size, show_confidence, show_advanced_charts):
    # Initialize session state for processed DataFrame
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'uploaded_file_hash' not in st.session_state:
        st.session_state.uploaded_file_hash = None

    if not api_key:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 2rem 0;">
                <h4 style="color: #856404; margin: 0;">🔐 API Key Required</h4>
                <p style="color: #856404; margin: 0.5rem 0 0 0;">
                    Please enter your Google Gemini API key in the sidebar to continue.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    if uploaded_file:
        current_file_hash = uploaded_file.name + str(uploaded_file.size)

    # Check if a new file has been uploaded or if no file is processed yet
    if uploaded_file and current_file_hash != st.session_state.uploaded_file_hash:
        st.session_state.processed_df = None # Clear previous data if new file
        st.session_state.uploaded_file_hash = current_file_hash

    if st.session_state.processed_df is None: # If no processed_df yet
        if uploaded_file is None:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                            padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 2rem 0;">
                    <h4 style="color: #856404; margin: 0;">📂 No File Uploaded</h4>
                    <p style="color: #856404; margin: 0.5rem 0 0 0;">
                        Please upload a file to proceed. Use the file uploader above to get started.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            return

        # Process the newly uploaded file
        try:
            original_df = pd.read_csv(uploaded_file)
            original_df.columns = original_df.columns.str.strip().str.lower() # Normalize column names
            if 'platform' not in original_df.columns:
                original_df['platform'] = 'Unknown'
                st.warning("⚠️ 'platform' column missing. Using 'Unknown' as default.")
            if "review" not in original_df.columns:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
                        <h4 style="color: #721c24; margin: 0;">❌ Column Error</h4>
                        <p style="color: #721c24; margin: 0.5rem 0 0 0;">
                            The uploaded file must contain a column named 'review'.<br>
                            <strong>Available columns:</strong> {", ".join(original_df.columns)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.stop()
            st.session_state.processed_df = original_df # Store for future reruns
            # original_df is now set for this run too
        except Exception as e:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 2rem; border-radius: 15px; border-left: 4px solid #dc3545;">
                    <h4 style="color: #721c24; margin-bottom: 0;">💥 Processing Error</h4>
                    <p style="color: #721c24; margin: 0.5rem 0 0 0;">An error occurred while processing your file: <code>{e}</code></p>
                    <p style="color: #721c24; margin: 0.5rem;0 0; font-size: 0.9rem;">Please check your file format and try again.</p>
                </div>
            """, unsafe_allow_html=True)
            logger.error(f"File processing error: {e}")
            st.session_state.processed_df = None # Clear on error
            return # Exit on error
    
    original_df = st.session_state.processed_df # Ensure original_df always points to the session state
    if original_df is None: # Exit if no DataFrame is available
        return

    try:
        st.markdown("""
            <div class="metric-card"; style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #0c5460;">📋 Data Preview</h3>
            </div>
        """, unsafe_allow_html=True)
        print(f"DEBUG: original_df columns at Data Preview: {original_df.columns.tolist()}")
        
        cols = st.columns(3)
        metrics = [
            ("Total Reviews", len(original_df), "#667eea"),
            ("Valid Reviews", original_df['review'].notna().sum(), "#28a745"),
            ("Avg. Rating", f"{original_df['rating'].mean():.1f}" if 'rating' in original_df.columns else "N/A", "#ffc107")
        ]
        
        for col, (label, value, color) in zip(cols, metrics):
            col.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: {color}; font-weight: 700;">{value}</div>
                        <div style="color: #6c757d;">{label}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        try:
            analyzer = SentimentAnalyzer(selected_model, api_key)
        except Exception as e:
            st.markdown(f"""
                <div class="metric-card"; style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
                    <h4 style="color: #721c24; margin: 0;">⚠️ Initialization Error</h4>
                    <p style="color: #721c24; margin: 0.5rem 0 0 0;">Failed to initialize analyzer: {e}</p>
                </div>
            """, unsafe_allow_html=True)
            return
        
        tag_suggestions = analyzer.suggest_tags_from_batch(original_df["review"].dropna().astype(str).tolist(), batch_size)
        
        if not tag_suggestions:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                            padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
                    <h4 style="color: #721c24; margin: 0;">❌ Tag Suggestion Failed</h4>
                    <p style="color: #721c24; margin: 0.5rem 0 0 0;">Gemini failed to generate tag suggestions. Please try again.</p>
                </div>
            """, unsafe_allow_html=True)
            return

        tag_html = " ".join([
            f"<span style='background-color:#f0f0f0; color:#333; padding:6px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.9rem;'>{tag}</span>"
            for tag in tag_suggestions
        ])
        
        st.markdown(
        f"""
        <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 1rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">✨ Recommended Tags</h3>
            {tag_html}
        </div>
        """, unsafe_allow_html=True)
        
        gb = GridOptionsBuilder.from_dataframe(original_df.head(10))
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
        grid_options = gb.build()
        AgGrid(original_df.head(10), gridOptions=grid_options, theme="alpine", height=300, allow_unsafe_jscode=True)
        
        _, col, _ = st.columns([1, 4, 1])
        with col:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">🤖 AI Processing in Progress</h3>
                        <p style="color: #6c757d;">Our advanced AI is analyzing your reviews...</p>
                    </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_container = st.empty()
                valid_reviews = []
                meta_info = []
                results = []
                total_reviews = len(original_df)
                start_time = time.time()
                
                for i, row in original_df.iterrows():
                    progress = (i + 1) / total_reviews
                    progress_bar.progress(progress)
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time / progress if progress > 0 else 0
                    remaining_time = max(0, estimated_total - elapsed_time)
                    
                    status_container.markdown(f"""
                        <div class="metric-card"; style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 1rem 0;">
                            <div style="font-size: 1.2rem; color: #667eea; font-weight: 600;">Processing Review {i + 1} of {total_reviews}</div>
                            <div style="color: #6c757d; margin-top: 0.5rem;">⏱️ Estimated time remaining: {remaining_time:.0f}s</div>
                            <div style="width: 100%; background: #e9ecef; border-radius: 10px; margin-top: 1rem;">
                                <div style="width: {progress*100:.1f}%; height: 8px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; transition: width 0.3s ease;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    review = row['review']
                    if pd.isna(review) or review.strip() == '':
                        results.append({
                            "review": review,
                            "platform": row.get('platform', 'Unknown'),
                            "rating": row.get('rating', 0),
                            "sentiment": "neutral",
                            "confidence": 0.0
                        })
                        continue
                    
                    valid_reviews.append(review)
                    meta_info.append({
                        "platform": row.get('platform', 'Unknown'),
                        "rating": row.get('rating', 0),
                        "review": review
                    })
            
                batch_results = analyzer.assign_tags_to_texts(valid_reviews, tag_suggestions, batch_size)
                
                for meta, analysis in zip(meta_info, batch_results):
                    result_entry = {
                        "review": meta["review"],
                        "platform": meta["platform"],
                        "rating": meta["rating"],
                        "tags": analysis.get("tags", [])
                    }
                    results.append(result_entry)
                
                progress_bar.empty()
                status_container.empty()
                
                processing_time = time.time() - start_time
                st.markdown(f"""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);">
                        <h2 style="color: #155724; margin-bottom: 1rem;">🎉 Analysis Complete!</h2>
                        <p style="color: #155724; font-size: 1.1rem;">
                            Successfully processed <strong>{total_reviews}</strong> reviews in <strong>{processing_time:.1f}</strong> seconds
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                results_df = pd.DataFrame(results)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Analysis Summary</h3>
                        <p style="color: #6c757d;">Key insights from your data</p>
                    </div>
                """, unsafe_allow_html=True)
                
                create_tag_beautiful_metrics(results_df)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📋 Detailed Results</h3>
                        <p style="color: #6c757d;">Detailed sentiment analysis results</p>
                    </div>
                """, unsafe_allow_html=True)
                
                gb = GridOptionsBuilder.from_dataframe(results_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
                grid_options = gb.build()
                
                AgGrid(results_df, gridOptions=grid_options, theme="alpine", height=400, fit_columns_on_grid_load=True, allow_unsafe_jscode=True)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Interactive Analytics Dashboard</h3>
                        <p style="color: #6c757d;">Visualize your sentiment analysis results</p>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(2)
                with cols[0]:
                    distribution_fig = create_tag_distribution_chart(results_df)
                    st.plotly_chart(distribution_fig, use_container_width=True)
                
                with cols[1]:
                    rating_fig = create_rating_chart(original_df)
                    st.plotly_chart(rating_fig, use_container_width=True)
                
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📥 Export Your Results</h3>
                        <p style="color: #6c757d;">Download your analysis in multiple formats</p>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(3)

                # === 1. CSV Export ===
                with cols[0]:
                    csv_output = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download CSV Report", data=csv_output,
                        file_name=f"tag_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", use_container_width=True
                    )

                # === 2. JSON Export (summary only) ===
                with cols[1]:
                    from collections import Counter

                    tag_counter = Counter(tag for tags in results_df["tags"] if isinstance(tags, list) for tag in tags)
                    tag_distribution = dict(tag_counter.most_common())

                    summary_stats = {
                        'analysis_summary': {
                            'total_reviews': total_reviews,
                            'processing_time_seconds': round(processing_time, 2),
                            'analysis_date': datetime.now().isoformat()
                        },
                        'tag_distribution': tag_distribution,
                        'tag_stats': {
                            'total_tags': sum(tag_counter.values()),
                            'unique_tags': len(tag_counter),
                            'average_tags_per_review': round(sum(tag_counter.values()) / total_reviews, 2)
                        }
                    }

                    json_output = json.dumps(summary_stats, indent=2, ensure_ascii=False).encode('utf-8')
                    st.download_button(
                        label="📋 Download JSON Summary", data=json_output,
                        file_name=f"tag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json", use_container_width=True
                    )

                # === 3. Excel Export (detailed + summary) ===
                with cols[2]:
                    try:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            # Sheet 1: Detailed result
                            results_df.to_excel(writer, sheet_name='Detailed Results', index=False)

                            # Sheet 2: Summary
                            summary_df = pd.DataFrame([
                                ['Total Reviews', total_reviews],
                                ['Processing Time (s)', round(processing_time, 2)],
                                ['Total Tags Assigned', sum(tag_counter.values())],
                                ['Unique Tags', len(tag_counter)],
                                ['Avg. Tags per Review', round(sum(tag_counter.values()) / total_reviews, 2)]
                            ], columns=['Metric', 'Value'])

                            summary_df.to_excel(writer, sheet_name='Summary', index=False)

                            # Sheet 3: Tag distribution
                            pd.DataFrame(tag_counter.most_common(), columns=["Tag", "Count"])\
                            .to_excel(writer, sheet_name="Tag Distribution", index=False)

                        st.download_button(
                            label="📈 Download Excel Report", data=excel_buffer.getvalue(),
                            file_name=f"tag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True
                        )

                    except ImportError:
                        st.markdown("""
                            <div class="metric-card"; style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
                                        padding: 1rem; border-radius: 10px; border-left: 1px solid #17a2b8;">
                                <p style="color: #0c5460;">📋 Install xlsxwriter for Excel export: <code>pip install xlsxwriter</code></p>
                            </div>
                        """, unsafe_allow_html=True)

                # === Final Thanks ===
                st.markdown("""
                    <div class="metric-card"; style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;
                                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);">
                        <h3 style="color: white; margin-bottom: 1rem;">🎊 Thank You for Using AI Sentiment Analysis Studio!</h3>
                        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem;">
                            Your tag analysis is complete. Feel free to analyze more data or download your results.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 2rem; border-radius: 15px; border-left: 4px solid #dc3545;">
                <h4 style="color: #721c24; margin-bottom: 0;">💥 Processing Error</h4>
                <p style="color: #721c24; margin: 0.5rem 0 0 0;">An error occurred while processing your file: <code>{e}</code></p>
                <p style="color: #721c24; margin: 0.5rem;0 0; font-size: 0.9rem;">Please check your file format and try again.</p>
            </div>
        """, unsafe_allow_html=True)
        logger.error(f"File processing error: {e}")

def main():
    """Main application function for sentiment analysis."""
    
    configure_page()
    apply_custom_css()
    create_animated_header()

    selected_page, api_key, selected_model, uploaded_file, batch_size, show_confidence, show_advanced_charts = create_beautiful_sidebar()
    
    if selected_page == "🏠 Home":
        home_page()

    if selected_page == "📊 Sentiment Analysis":
        sentiment_page(api_key, selected_model, uploaded_file, batch_size, show_confidence, show_advanced_charts)
        
    if selected_page == "🛠️ Tag Classification":
        classify_page(api_key, selected_model, uploaded_file, batch_size, show_confidence, show_advanced_charts)
    
if __name__ == "__main__":
    main()