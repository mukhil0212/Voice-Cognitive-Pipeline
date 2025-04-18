import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List

import sys
import os

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_processor import AudioProcessor
from transcriber import Transcriber
from analyzer import SpeechAnalyzer
from word_recall import WordRecallDetector, NamingTaskAnalyzer
from sentence_completion import SentenceCompletionDetector

class Dashboard:
    def __init__(self, groq_api_key: str):
        self.audio_processor = AudioProcessor()
        self.transcriber = Transcriber(groq_api_key)
        self.analyzer = SpeechAnalyzer()

        # Set default naming task targets
        self.transcriber.set_naming_targets([
            "apple", "banana", "car", "dog", "elephant",
            "flower", "guitar", "house", "ice cream", "jacket"
        ])

    def analyze_audio(self, audio_path: str, baseline_text: str = None) -> Dict:
        """Analyze a single audio file."""
        # Process audio
        y, sr = self.audio_processor.load_and_preprocess(audio_path)

        # Extract features
        pauses = self.audio_processor.extract_pauses(y, sr)
        pitch_stats = self.audio_processor.compute_pitch_stats(y, sr)

        # Transcribe
        transcript = self.transcriber.transcribe(audio_path)

        # Analyze transcript
        transcript_analysis = self.transcriber.analyze_transcript(transcript, baseline_text)

        # Compute speech rate
        duration = len(y) / sr
        speech_rate = self.audio_processor.compute_speech_rate(
            transcript_analysis['word_count'],
            duration
        )

        # Calculate cognitive risk score
        risk_score = self.transcriber.get_cognitive_risk_score(transcript_analysis)

        # Compile results
        results = {
            'file': Path(audio_path).name,
            'transcript': transcript,
            'pause_count': len(pauses),
            'hesitation_count': transcript_analysis['hesitation_count'],
            'speech_rate': speech_rate,
            'pitch_stats': pitch_stats,
            'word_recall': transcript_analysis.get('word_recall', {}),
            'naming_task': transcript_analysis.get('naming_task', {}),
            'sentence_completion': transcript_analysis.get('sentence_completion', {}),
            'completion_score': transcript_analysis.get('completion_score', 0),
            'cognitive_risk_score': risk_score,
            'cognitive_assessment': {
                'risk_level': 'Low' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'High',
                'indicators': {
                    'hesitation_frequency': 'Normal' if transcript_analysis.get('hesitation_rate', 0) < 0.1 else 'Elevated',
                    'speech_rate': 'Normal' if 80 <= speech_rate <= 160 else 'Abnormal',
                    'sentence_structure': 'Normal' if transcript_analysis.get('completion_score', 1) > 0.8 else 'Impaired'
                }
            }
        }

        return results

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.title("Speech Analysis Dashboard")

        # File upload
        uploaded_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])

        # Baseline text input
        baseline_text = st.text_area("Baseline Text (for word recall analysis)", "")

        if uploaded_file and st.button("Analyze"):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                # Analyze audio
                results = self.analyze_audio(temp_path, baseline_text if baseline_text else None)

                # Display results
                st.header("Analysis Results")

                # Transcript
                st.subheader("Transcript")
                st.write(results['transcript'])

                # Basic Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Speech Rate (words/min)", f"{results['speech_rate']:.1f}")
                with col2:
                    st.metric("Pause Count", results['pause_count'])
                with col3:
                    st.metric("Hesitation Count", results['hesitation_count'])

                # Word Recall Analysis
                if 'word_recall' in results:
                    st.subheader("Word Recall Analysis")
                    recall = results['word_recall']
                    st.write(f"Similarity Score: {recall['similarity_score']:.2f}")
                    if recall['substitutions']:
                        st.write("Substitutions:")
                        for sub in recall['substitutions']:
                            st.write(f"- Expected: '{sub['expected']}', Said: '{sub['actual']}'")

                # Sentence Completion
                if 'sentence_completion' in results:
                    st.subheader("Sentence Completion Analysis")
                    completion = results['sentence_completion']
                    st.write(f"Completion Score: {results['completion_score']:.2f}")
                    st.write(f"Complete Sentences: {len(completion['complete_sentences'])}")
                    st.write(f"Incomplete Sentences: {len(completion['incomplete_sentences'])}")

                # Naming Task
                if 'naming_task' in results:
                    st.subheader("Naming Task Analysis")
                    naming = results['naming_task']
                    st.write(f"Success Rate: {naming['success_rate']:.2f}")
                    if naming['failed_attempts']:
                        st.write("Failed Attempts:")
                        for attempt in naming['failed_attempts']:
                            st.write(f"- Target: '{attempt['target']}', Said: '{attempt['actual']}'")

                # Cognitive Risk Assessment
                st.header("Cognitive Risk Assessment")
                risk_score = results['cognitive_risk_score']
                risk_level = results['cognitive_assessment']['risk_level']
                indicators = results['cognitive_assessment']['indicators']

                # Create columns for risk metrics
                col1, col2 = st.columns(2)
                with col1:
                    # Create a gauge chart for risk score
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.set_style("whitegrid")
                    ax.barh([0], [risk_score], color='g' if risk_score > 0.7 else 'y' if risk_score > 0.4 else 'r')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlabel("Risk Score (higher is better)")
                    ax.set_yticks([])
                    # Add text annotation for the score
                    ax.text(risk_score, 0, f"{risk_score:.2f}", va='center', ha='center',
                            fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8))
                    st.pyplot(fig)

                with col2:
                    # Display risk level and indicators
                    st.subheader(f"Risk Level: {risk_level}")
                    st.write("Indicators:")
                    for indicator, status in indicators.items():
                        indicator_name = indicator.replace('_', ' ').title()
                        color = "green" if status == "Normal" else "red"
                        st.markdown(f"- {indicator_name}: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

                # Add ML-based analysis section
                st.header("Machine Learning Analysis")
                st.write("To perform anomaly detection and clustering, upload multiple audio samples.")

                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Speech Patterns", "Pitch Analysis", "Cognitive Metrics"])

                with tab1:
                    # Create a bar chart for speech patterns
                    fig, ax = plt.subplots(figsize=(10, 5))
                    metrics = ['speech_rate', 'pause_count', 'hesitation_count']
                    values = [results['speech_rate'], results['pause_count'], results['hesitation_count']]
                    colors = ['skyblue', 'lightgreen', 'salmon']
                    ax.bar(metrics, values, color=colors)
                    ax.set_title("Speech Pattern Metrics")
                    ax.set_ylabel("Value")
                    # Add value labels on top of bars
                    for i, v in enumerate(values):
                        ax.text(i, v + 0.1, f"{v:.1f}" if isinstance(v, float) else f"{v}",
                                ha='center', va='bottom')
                    st.pyplot(fig)

                with tab2:
                    # Create a visualization for pitch analysis
                    fig, ax = plt.subplots(figsize=(10, 5))
                    pitch_data = [
                        results['pitch_stats']['pitch_mean'],
                        results['pitch_stats']['pitch_std'],
                        results['pitch_stats']['pitch_range']
                    ]
                    pitch_labels = ['Mean', 'Standard Deviation', 'Range']
                    ax.bar(pitch_labels, pitch_data, color='purple')
                    ax.set_title("Pitch Analysis (Hz)")
                    ax.set_ylabel("Frequency (Hz)")
                    # Add value labels
                    for i, v in enumerate(pitch_data):
                        ax.text(i, v + 0.1, f"{v:.1f}", ha='center', va='bottom')
                    st.pyplot(fig)

                with tab3:
                    # Create a visualization for cognitive metrics
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # Extract cognitive metrics
                    word_recall_score = results['word_recall'].get('similarity_score', 0) if results['word_recall'] else 0
                    naming_score = results['naming_task'].get('success_rate', 0) if results['naming_task'] else 0
                    completion_score = results['completion_score']

                    metrics = ['Word Recall', 'Naming Task', 'Sentence Completion', 'Overall Risk Score']
                    values = [word_recall_score, naming_score, completion_score, risk_score]
                    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

                    ax.bar(metrics, values, color=colors)
                    ax.set_title("Cognitive Assessment Metrics")
                    ax.set_ylabel("Score (0-1)")
                    ax.set_ylim(0, 1)
                    # Add value labels
                    for i, v in enumerate(values):
                        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
                    st.pyplot(fig)

            finally:
                # Cleanup
                Path(temp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    # Get Groq API key from environment or Streamlit secrets
    import os
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable is not set. Please set it before running the dashboard.")
        st.stop()

    dashboard = Dashboard(groq_api_key)
    dashboard.run_dashboard()
