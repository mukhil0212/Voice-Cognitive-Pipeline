from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tempfile
import os
import sys
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import datetime as import_datetime
from typing import Dict, List, Optional
import json

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_processor import AudioProcessor
from transcriber import Transcriber
from analyzer import SpeechAnalyzer

# Global variable to store analysis results for visualization and batch processing
analysis_results = []

app = FastAPI(title="Speech Analysis API",
         description="API for analyzing speech patterns to detect cognitive decline indicators")

# Initialize components
audio_processor = AudioProcessor()

# Get Groq API key from environment
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the API.")

transcriber = Transcriber(groq_api_key)
analyzer = SpeechAnalyzer()

@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...), baseline_text: str = None, naming_targets: str = None):
    """Analyze uploaded audio file and return comprehensive speech analysis.

    Parameters:
    - file: Audio file (WAV/MP3)
    - baseline_text: Optional reference text for word recall analysis
    - naming_targets: Optional comma-separated list of target words for naming task

    Returns JSON with:
    - Transcription
    - Pause and hesitation analysis
    - Speech rate and pitch statistics
    - Word recall performance (if baseline_text provided)
    - Naming task results (if naming_targets provided)
    - Sentence completion analysis
    - Cognitive risk score and assessment
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Process audio
        y, sr = audio_processor.load_and_preprocess(temp_path)

        # Extract features
        pauses = audio_processor.extract_pauses(y, sr)
        pitch_stats = audio_processor.compute_pitch_stats(y, sr)

        # Set naming targets if provided
        if naming_targets:
            target_words = [word.strip() for word in naming_targets.split(',')]
            transcriber.set_naming_targets(target_words)

        # Transcribe
        transcript = transcriber.transcribe(temp_path)
        transcript_analysis = transcriber.analyze_transcript(transcript, baseline_text)

        # Compute speech rate
        duration = len(y) / sr
        speech_rate = audio_processor.compute_speech_rate(
            transcript_analysis['word_count'],
            duration
        )

        # Calculate cognitive risk score
        risk_score = transcriber.get_cognitive_risk_score(transcript_analysis)

        # Compile results in the exact format requested in the feedback
        results = {
            'features': {
                'pause_count': len(pauses),
                'hesitation_count': transcript_analysis['hesitation_count'],
                'speech_rate': speech_rate,
                'pitch_stats': pitch_stats,
                'transcript': transcript,
                'word_recall': transcript_analysis.get('word_recall', {}),
                'naming_task': transcript_analysis.get('naming_task', {}),
                'sentence_completion': transcript_analysis.get('sentence_completion', {}),
            },
            'cluster_label': None,  # Will be populated if we have enough samples
            'anomaly_score': None,  # Will be populated if we have enough samples
            'risk_score': risk_score,  # Renamed from cognitive_risk_score to match requested format
            'cognitive_assessment': {
                'risk_level': 'Low' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'High',
                'indicators': {
                    'hesitation_frequency': 'Normal' if transcript_analysis.get('hesitation_rate', 0) < 0.1 else 'Elevated',
                    'speech_rate': 'Normal' if 80 <= speech_rate <= 160 else 'Abnormal',
                    'sentence_structure': 'Normal' if transcript_analysis.get('completion_score', 1) > 0.8 else 'Impaired'
                }
            },
            'timestamp': import_datetime.datetime.now().isoformat()
        }

        # If we have enough samples, perform anomaly detection

        # Create a feature dictionary for ML analysis and visualization
        ml_features = {
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

        # Store result for visualization and batch analysis
        analysis_results.append(ml_features)

        # If we have at least 2 samples, we can perform clustering
        if len(analysis_results) >= 2:
            # Perform anomaly detection on all samples
            feature_matrix = analyzer.prepare_features([ml_features])
            all_features = analyzer.prepare_features(analysis_results)
            labels, anomaly_scores = analyzer.fit_predict(all_features)

            # Get the label and score for the current sample (last one added)
            current_label = labels[-1]
            current_anomaly_score = anomaly_scores[-1]

            # Add ML results to the response
            results['cluster_label'] = int(current_label)
            results['anomaly_score'] = float(current_anomaly_score)
            results['is_anomaly'] = bool(current_label == -1)
        else:
            # Not enough samples for clustering yet
            results['cluster_label'] = None
            results['anomaly_score'] = None
            results['is_anomaly'] = None

        return results

    finally:
        # Cleanup
        os.unlink(temp_path)

# Debug endpoints to check and manage the state of analysis_results
@app.get("/debug")
async def debug_results():
    """Debug endpoint to check the state of analysis_results."""
    return {
        "count": len(analysis_results),
        "results": analysis_results
    }

@app.get("/reset")
async def reset_results():
    """Reset the analysis_results list."""
    global analysis_results
    analysis_results = []
    return {"message": "Analysis results reset successfully", "count": 0}

@app.post("/analyze-batch/")
async def analyze_batch(files: List[UploadFile] = File(...), baseline_text: str = None, naming_targets: str = None):
    """Analyze multiple audio files and return comprehensive batch analysis.

    Parameters:
    - files: List of audio files (WAV/MP3)
    - baseline_text: Optional reference text for word recall analysis
    - naming_targets: Optional comma-separated list of target words for naming task

    Returns JSON with:
    - Individual analysis for each file
    - Batch summary statistics
    - Anomaly detection results
    - Cluster analysis
    """
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files provided"})

    # Set naming targets if provided
    if naming_targets:
        target_words = [word.strip() for word in naming_targets.split(',')]
        transcriber.set_naming_targets(target_words)

    batch_results = []
    temp_files = []

    try:
        # Process each file
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_files.append(temp_path)

            # Process audio
            y, sr = audio_processor.load_and_preprocess(temp_path)

            # Extract features
            pauses = audio_processor.extract_pauses(y, sr)
            pitch_stats = audio_processor.compute_pitch_stats(y, sr)

            # Transcribe
            transcript = transcriber.transcribe(temp_path)
            transcript_analysis = transcriber.analyze_transcript(transcript, baseline_text)

            # Compute speech rate
            duration = len(y) / sr
            speech_rate = audio_processor.compute_speech_rate(
                transcript_analysis['word_count'],
                duration
            )

            # Calculate cognitive risk score
            risk_score = transcriber.get_cognitive_risk_score(transcript_analysis)

            # Create feature dictionary
            features = {
                'filename': file.filename,
                'pause_count': len(pauses),
                'hesitation_count': transcript_analysis['hesitation_count'],
                'speech_rate': speech_rate,
                'pitch_stats': pitch_stats,
                'transcript': transcript,
                'word_recall': transcript_analysis.get('word_recall', {}),
                'naming_task': transcript_analysis.get('naming_task', {}),
                'sentence_completion': transcript_analysis.get('sentence_completion', {}),
                'completion_score': transcript_analysis.get('completion_score', 0),
                'cognitive_risk_score': risk_score
            }

            batch_results.append(features)

            # Also add to the global analysis_results for visualization
            analysis_results.append(features)

        # Perform batch analysis using the SpeechAnalyzer
        batch_analysis = analyzer.analyze_batch(batch_results)

        return batch_analysis

    finally:
        # Cleanup temporary files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass

@app.get("/visualize")
async def visualize_results():
    """Generate visualizations of analysis results."""
    if not analysis_results:
        return HTMLResponse("<h1>No analysis results available</h1><p>Upload audio files first to generate visualizations.</p>")

    # Create visualizations with more detailed plots
    plt.figure(figsize=(15, 15))

    # Extract data for plotting
    sample_ids = [f"Sample {i+1}" for i in range(len(analysis_results))]
    speech_rates = [result['speech_rate'] for result in analysis_results]
    pause_counts = [result['pause_count'] for result in analysis_results]
    hesitation_counts = [result['hesitation_count'] for result in analysis_results]
    risk_scores = [result['cognitive_risk_score'] for result in analysis_results]

    # Extract pitch statistics
    pitch_means = [result['pitch_stats']['pitch_mean'] for result in analysis_results]
    pitch_stds = [result['pitch_stats']['pitch_std'] for result in analysis_results]

    # Extract completion scores if available
    completion_scores = [result.get('completion_score', 0) for result in analysis_results]

    # Get risk levels
    risk_levels = [result['cognitive_assessment']['risk_level'] for result in analysis_results]
    risk_level_colors = ['green' if level == 'Low' else 'orange' if level == 'Medium' else 'red' for level in risk_levels]

    # Plot 1: Speech Rate
    plt.subplot(3, 2, 1)
    bars = plt.bar(sample_ids, speech_rates, color='skyblue')
    plt.title('Speech Rate (words/min)', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Words per minute')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')

    # Plot 2: Pause Count
    plt.subplot(3, 2, 2)
    bars = plt.bar(sample_ids, pause_counts, color='lightgreen')
    plt.title('Pause Count', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Number of pauses')
    plt.xticks(rotation=45)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')

    # Plot 3: Hesitation Count
    plt.subplot(3, 2, 3)
    bars = plt.bar(sample_ids, hesitation_counts, color='salmon')
    plt.title('Hesitation Markers', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Number of hesitations')
    plt.xticks(rotation=45)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')

    # Plot 4: Cognitive Risk Score with color-coded risk levels
    plt.subplot(3, 2, 4)
    bars = plt.bar(sample_ids, risk_scores, color=risk_level_colors)
    plt.title('Cognitive Risk Score (higher is better)', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')

    # Plot 5: Pitch Variability
    plt.subplot(3, 2, 5)
    x = np.arange(len(sample_ids))
    width = 0.35
    plt.bar(x - width/2, pitch_means, width, label='Mean Pitch', color='lightblue')
    plt.bar(x + width/2, pitch_stds, width, label='Pitch Variability', color='darkblue')
    plt.title('Pitch Analysis', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(x, sample_ids, rotation=45)
    plt.legend()

    # Plot 6: Sentence Completion Score
    plt.subplot(3, 2, 6)
    bars = plt.bar(sample_ids, completion_scores, color='purple')
    plt.title('Sentence Completion Score', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Create HTML response with embedded image and data table
    html_content = f"""
    <html>
        <head>
            <title>Speech Analysis Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .visualization {{ margin-top: 30px; text-align: center; }}
                .back-link {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Speech Analysis Visualizations</h1>
                <p>Showing analysis results for {len(analysis_results)} audio samples</p>

                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3>Summary Statistics</h3>
                    <div style="display: flex; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px; margin: 10px;">
                            <h4>Speech Patterns</h4>
                            <p>Average Speech Rate: <strong>{sum([r['speech_rate'] for r in analysis_results]) / len(analysis_results):.2f}</strong> wpm</p>
                            <p>Average Pause Count: <strong>{sum([r['pause_count'] for r in analysis_results]) / len(analysis_results):.1f}</strong></p>
                            <p>Average Hesitation Count: <strong>{sum([r['hesitation_count'] for r in analysis_results]) / len(analysis_results):.1f}</strong></p>
                        </div>
                        <div style="flex: 1; min-width: 200px; margin: 10px;">
                            <h4>Cognitive Assessment</h4>
                            <p>Average Risk Score: <strong>{sum([r.get('cognitive_risk_score', 0) for r in analysis_results]) / len(analysis_results):.2f}</strong></p>
                            <p>Low Risk Samples: <strong>{sum(1 for r in analysis_results if r.get('cognitive_assessment', {}).get('risk_level') == 'Low')}</strong></p>
                            <p>Medium Risk Samples: <strong>{sum(1 for r in analysis_results if r.get('cognitive_assessment', {}).get('risk_level') == 'Medium')}</strong></p>
                            <p>High Risk Samples: <strong>{sum(1 for r in analysis_results if r.get('cognitive_assessment', {}).get('risk_level') == 'High')}</strong></p>
                        </div>
                        <div style="flex: 1; min-width: 200px; margin: 10px;">
                            <h4>Anomaly Detection</h4>
                            <p>Anomalies Detected: <strong>{sum(1 for r in analysis_results if r.get('cluster_label') == -1)}</strong></p>
                            <p>Clusters Found: <strong>{len(set([r.get('cluster_label') for r in analysis_results if r.get('cluster_label') is not None and r.get('cluster_label') != -1]))}</strong></p>
                            <p>Average Anomaly Score: <strong>{sum([r.get('anomaly_score', 0) for r in analysis_results if isinstance(r.get('anomaly_score'), (int, float))]) / sum(1 for r in analysis_results if isinstance(r.get('anomaly_score'), (int, float))) if sum(1 for r in analysis_results if isinstance(r.get('anomaly_score'), (int, float))) > 0 else 0:.2f}</strong></p>
                        </div>
                    </div>
                </div>

                <div class="visualization">
                    <img src="data:image/png;base64,{img_str}" alt="Speech Analysis Visualizations">
                </div>

                <h2>Analysis Results</h2>
                <table>
                    <tr>
                        <th>Sample</th>
                        <th>Speech Rate (wpm)</th>
                        <th>Pause Count</th>
                        <th>Hesitation Count</th>
                        <th>Pitch Mean (Hz)</th>
                        <th>Pitch Std (Hz)</th>
                        <th>Risk Score</th>
                        <th>Risk Level</th>
                        <th>Anomaly Score</th>
                        <th>Cluster</th>
                    </tr>
    """

    for i, result in enumerate(analysis_results):
        # Determine risk level based on cognitive risk score
        risk_score = result.get('cognitive_risk_score', 0)
        risk_level = 'Low' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'High'

        # Get risk level from cognitive_assessment if available
        if 'cognitive_assessment' in result and 'risk_level' in result['cognitive_assessment']:
            risk_level = result['cognitive_assessment']['risk_level']

        # Get pitch statistics
        pitch_mean = result['pitch_stats']['pitch_mean']
        pitch_std = result['pitch_stats']['pitch_std']

        # Get anomaly score and cluster label if available
        anomaly_score = result.get('anomaly_score', 'N/A')
        if isinstance(anomaly_score, float):
            anomaly_score = f"{anomaly_score:.2f}"

        cluster_label = result.get('cluster_label', 'N/A')
        if cluster_label == -1:
            cluster_label = "Outlier"

        # Set row color based on risk level
        row_color = "#e6ffe6" if risk_level == "Low" else "#fff2e6" if risk_level == "Medium" else "#ffe6e6"

        html_content += f"""
                    <tr style="background-color: {row_color};">
                        <td>{i+1}</td>
                        <td>{result['speech_rate']:.2f}</td>
                        <td>{result['pause_count']}</td>
                        <td>{result['hesitation_count']}</td>
                        <td>{pitch_mean:.2f}</td>
                        <td>{pitch_std:.2f}</td>
                        <td>{risk_score:.2f}</td>
                        <td>{risk_level}</td>
                        <td>{anomaly_score}</td>
                        <td>{cluster_label}</td>
                    </tr>
        """

    html_content += """
                </table>

                <div class="back-link">
                    <a href="/">Back to Upload Form</a>
                </div>
            </div>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)

@app.get("/")
async def root():
    """Return simple HTML form for testing."""
    return HTMLResponse("""
        <html>
            <head>
                <title>Speech Analysis for Cognitive Assessment</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                    .container { max-width: 800px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; }
                    .info { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Speech Analysis for Cognitive Assessment</h1>
                    <div class="info">
                        <p>This tool analyzes speech patterns to detect potential cognitive decline indicators.</p>
                        <p>Features analyzed include: pauses, hesitations, word recall, speech rate, pitch variability, and sentence completion.</p>
                    </div>
                    <form action="/analyze/" enctype="multipart/form-data" method="post">
                    <h2>Speech Analysis Tool</h2>
                    <p>Upload an audio file for cognitive assessment</p>
                    <div style="margin-bottom: 15px;">
                        <label for="file">Audio File (WAV/MP3):</label><br>
                        <input name="file" id="file" type="file" accept=".wav,.mp3">
                    </div>
                    <div style="margin-bottom: 15px;">
                        <label for="baseline_text">Baseline Text (for word recall analysis):</label><br>
                        <textarea name="baseline_text" id="baseline_text" rows="3" cols="50" placeholder="Enter text that the speaker was supposed to recall..."></textarea>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <label for="naming_targets">Naming Task Targets (comma-separated):</label><br>
                        <input name="naming_targets" id="naming_targets" type="text" style="width: 100%;" placeholder="apple, banana, car, dog, elephant">
                    </div>
                    <input type="submit" value="Analyze Speech" style="padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer;">
                </form>

                <div style="margin-top: 20px;">
                    <a href="/visualize" style="padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; display: inline-block; margin-right: 10px;">View Analysis Visualizations</a>
                    <a href="/reset" style="padding: 10px 20px; background-color: #e74c3c; color: white; text-decoration: none; display: inline-block;">Reset All Results</a>
                </div>

                <h2 style="margin-top: 30px;">Batch Analysis</h2>
                <p>Upload multiple audio files for comparative analysis</p>

                <form action="/analyze-batch/" enctype="multipart/form-data" method="post">
                    <div style="margin-bottom: 15px;">
                        <label for="batch_files">Audio Files (WAV/MP3):</label><br>
                        <input name="files" id="batch_files" type="file" multiple accept=".wav,.mp3">
                    </div>
                    <div style="margin-bottom: 15px;">
                        <label for="batch_baseline_text">Baseline Text (for word recall analysis):</label><br>
                        <textarea name="baseline_text" id="batch_baseline_text" rows="3" cols="50" placeholder="Enter text that the speakers were supposed to recall..."></textarea>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <label for="batch_naming_targets">Naming Task Targets (comma-separated):</label><br>
                        <input name="naming_targets" id="batch_naming_targets" type="text" style="width: 100%;" placeholder="apple, banana, car, dog, elephant">
                    </div>
                    <input type="submit" value="Analyze Batch" style="padding: 10px 20px; background-color: #9b59b6; color: white; border: none; cursor: pointer;">
                </form>
                </div>
            </body>
        </html>
    """)
