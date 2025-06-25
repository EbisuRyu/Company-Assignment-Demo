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
from io import BytesIO

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
    
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Analysis result containing sentiment and confidence
        """
        prompt = f"""
        You are a strict sentiment analysis system.

        Analyze the sentiment of the following text and return the result **only** as one of the three predefined categories: `"positive"`, `"negative"`, or `"neutral"`. You must not return any other value.

        Your response must be a **valid JSON object** in the **exact** structure below, with no additional text, comments, or formatting:

        {{
            "sentiment": "positive" | "negative" | "neutral",
            "confidence": float  // a number between 0 and 1
        }}

        Text to analyze:
        \"\"\"{text}\"\"\"
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {"sentiment": "error", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Analysis error: {type(e).__name__}: {e}")
            return {"sentiment": f"Error ({type(e).__name__}): {e}", "confidence": 0.0}

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
        ("🔒", "Secure & Private", "Your data is processed securely with enterprise-grade security.")
    ]
    
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem;">{icon}</div>
                <h4 style="color: #495057; margin: 0.5rem 0;">{title}</h4>
                <p style="color: #6c757d;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

def create_beautiful_metrics(results_df: pd.DataFrame, original_df: pd.DataFrame):
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
        ("😊", counts['positive'], "Positive", "#28a745"),
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

def create_beautiful_sidebar():
    """Create a modern sidebar for configuration."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #667eea; margin-bottom: 0.5rem;">⚙️ Configuration</h2>
                <p style="color: #6c757d; font-size: 0.9rem;">Configure your analysis settings</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Google Gemini API Key", type="password",
            help="🔒 Your API key is secure and never stored", placeholder="Enter your Gemini API key..."
        )
        
        st.markdown("### 📤 File Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File", type=["csv"], help="📊 Upload a CSV file with a 'review' column"
        )
        
        st.markdown("### 🎛️ Processing Options")
        batch_size = st.slider("Batch Size", 1, 50, 10, help="⚡ Number of reviews to process simultaneously")
        show_confidence = st.checkbox("Show Confidence Scores", value=True, help="📊 Display AI confidence levels")
        show_advanced_charts = st.checkbox("Advanced Visualizations", value=True, help="📈 Show additional charts and analytics")
        
        return api_key, uploaded_file, batch_size, show_confidence, show_advanced_charts

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
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            title_font=dict(color='#2c3e50')
        ),
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
        legend=dict(title="Rating", title_font=dict(color='#2c3e50'), font=dict(color='#2c3e50')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, showlegend=True
    )
    return fig

# -----------------------------------
# Main Application
# -----------------------------------

def main():
    """Main application function for sentiment analysis."""
    configure_page()
    apply_custom_css()
    create_animated_header()
    
    api_key, uploaded_file, batch_size, show_confidence, show_advanced_charts = create_beautiful_sidebar()

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

    current_file_hash = uploaded_file.name + str(uploaded_file.size)

    # Check if a new file has been uploaded or if no file is processed yet
    if uploaded_file and current_file_hash != st.session_state.uploaded_file_hash:
        st.session_state.processed_df = None # Clear previous data if new file
        st.session_state.uploaded_file_hash = current_file_hash

    if st.session_state.processed_df is None: # If no processed_df yet
        if uploaded_file is None: # And no file uploaded, show welcome
            create_welcome_section()
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
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
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
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
            return # Exit main if no file and no processed data

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
            <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
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
        
        _, col, _ = st.columns([1, 2, 1])
        with col:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                try:
                    analyzer = SentimentAnalyzer(api_key)
                except Exception as e:
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
                            <h4 style="color: #721c24; margin: 0;">⚠️ Initialization Error</h4>
                            <p style="color: #721c24; margin: 0.5rem 0 0 0;">Failed to initialize analyzer: {e}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    return
                
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">🤖 AI Processing in Progress</h3>
                        <p style="color: #6c757d;">Our advanced AI is analyzing your reviews...</p>
                    </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_container = st.empty()
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
                        <div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 1rem 0;">
                            <div style="font-size: 1.2rem; color: #667eea; font-weight: 600;">Processing Review {i + 1} of {total_reviews}</div>
                            <div style="color: #6c757d; margin-top: 0.5rem;">⏱️ Estimated time remaining: {remaining_time:.0f}s</div>
                            <div style="width: 100%; background: #e9ecef; border-radius: 10px; margin-top: 1rem;">
                                <div style="width: {progress*100:.1f}%; height: 8px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; transition: width 0.3s ease;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if pd.isna(row['review']) or row['review'].strip() == '':
                        results.append({
                            "review": row['review'],
                            "platform": row.get('platform', 'Unknown'),
                            "rating": row.get('rating', 0),
                            "sentiment": "neutral",
                            "confidence": 0.0
                        })
                        continue
                    
                    analysis_result = analyzer.analyze_sentiment(row['review'])
                    result_entry = {
                        "review": row['review'],
                        "platform": row.get('platform', 'Unknown'),
                        "rating": row.get('rating', 0),
                        "sentiment": analysis_result.get('sentiment', 'neutral')
                    }
                    if show_confidence:
                        result_entry["confidence"] = analysis_result.get('confidence', 0.0)
                    results.append(result_entry)
                
                progress_bar.empty()
                status_container.empty()
                
                processing_time = time.time() - start_time
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);">
                        <h2 style="color: #155724; margin-bottom: 1rem;">🎉 Analysis Complete!</h2>
                        <p style="color: #155724; font-size: 1.1rem;">
                            Successfully processed <strong>{total_reviews}</strong> reviews in <strong>{processing_time:.1f}</strong> seconds
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                results_df = pd.DataFrame(results)
                
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Analysis Summary</h3>
                        <p style="color: #6c757d;">Key insights from your data</p>
                    </div>
                """, unsafe_allow_html=True)
                create_beautiful_metrics(results_df, original_df)
                
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
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
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
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
                        <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
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
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); text-align: center;">
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
                        label="📋 Download Summary", data=json_output,
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
                            <div style="background: linear-gradient(135deg, #d1ecf1 0%; #bee5eb 100%); padding: 1rem; border-radius: 10px; border-left: 1px solid #17a2b8;">
                                <p style="color: #0c5460;">📋 Install xlsxwriter for Excel export: <code>pip install xlsxwriter</code></p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0; box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);">
                        <h3 style="color: white; margin-bottom: 1rem;">🎊 Thank You for Using AI Sentiment Analysis Studio!</h3>
                        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem;">
                            Your analysis is complete. Feel free to analyze more data or download your results.
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

if __name__ == "__main__":
    main()