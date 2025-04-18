# Speech Analysis Pipeline

A comprehensive Python pipeline for analyzing speech patterns in audio recordings, with advanced cognitive assessment features. The pipeline performs audio preprocessing, transcription using Groq's Whisper model, and extensive linguistic analysis to identify potential cognitive issues.

## Features

### Audio Processing
- Resampling to 16 kHz
- Silence trimming and volume normalization
- High-quality transcription using Groq's Whisper Large V3 Turbo model

### Linguistic Analysis
- Pause detection and analysis
- Hesitation marker detection (um, uh, er)
- Speech rate calculation
- Pitch variability analysis using Parselmouth

### Cognitive Assessment
- Word recall analysis with substitution detection
- Naming task performance evaluation
- Sentence completion analysis using spaCy
- Overall cognitive risk scoring

### Visualization & Reporting
- Interactive Streamlit dashboard
- PCA and t-SNE visualizations
- FastAPI endpoint for real-time analysis
- Detailed HTML/PDF reports

## Setup

### Prerequisites

1. Python 3.11 or later (but less than 3.13)
2. pip (package installer for Python)
3. Git
4. ffmpeg (for audio processing)

#### Installing ffmpeg

**On macOS:**
```bash
brew install ffmpeg
```

**On Windows:**
1. Download the latest static build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the ZIP file
3. Add the `bin` folder to your system PATH:
   - Right-click on 'This PC' or 'My Computer' and select 'Properties'
   - Click on 'Advanced system settings'
   - Click on 'Environment Variables'
   - Under 'System variables', find and select 'Path', then click 'Edit'
   - Click 'New' and add the path to the ffmpeg bin folder (e.g., `C:\ffmpeg\bin`)
   - Click 'OK' on all dialogs
4. Verify installation by opening a new Command Prompt and typing:
   ```
   ffmpeg -version
   ```

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Virtual Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd speech_analysis
```

2. Create a virtual environment:

**On macOS/Linux:**
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

You'll know the virtual environment is activated when you see `(venv)` at the beginning of your terminal prompt.

3. Upgrade pip and install build tools:
```bash
pip install --upgrade pip setuptools wheel
```

4. Install project dependencies:
```bash
pip install -r requirements.txt
```

5. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

6. Set up environment variables:

**On macOS/Linux:**
```bash
export GROQ_API_KEY="your_api_key_here"
```
You can add this to your `~/.bashrc` or `~/.zshrc` file to make it permanent.

**On Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=your_api_key_here
```

**On Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "your_api_key_here"
```

**For permanent environment variables on Windows:**
1. Right-click on 'This PC' or 'My Computer' and select 'Properties'
2. Click on 'Advanced system settings'
3. Click on 'Environment Variables'
4. Under 'User variables', click 'New'
5. Enter 'GROQ_API_KEY' as the variable name and your API key as the value
6. Click 'OK' on all dialogs

7. Place audio files in `data/audio/` directory (supported formats: WAV, MP3)

### Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

### Troubleshooting

If you encounter any issues during installation:

#### Python Version
Make sure you're using Python 3.11:
```bash
python --version  # On Windows
python3 --version  # On macOS/Linux
```

#### Permission Errors
If you get permission errors:
```bash
pip install --user -r requirements.txt
```

#### Package Installation Issues
If you have issues with specific packages, try installing them individually:
```bash
pip install librosa
pip install groq
# etc.
```

#### Windows-Specific Issues

1. **Microsoft Visual C++ Build Tools**
   Some packages require Microsoft Visual C++ Build Tools. If you get errors about missing compilers:
   - Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select "Desktop development with C++"

2. **Path Too Long Errors**
   If you get "path too long" errors on Windows:
   - Enable long path support by running this in PowerShell as Administrator:
     ```powershell
     New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
     ```
   - Restart your computer

3. **Audio Playback Issues**
   If audio playback doesn't work in Streamlit:
   - Install PyAudio: `pip install pyaudio`
   - If PyAudio installation fails, download the appropriate wheel file from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install it with:
     ```
     pip install C:\path\to\downloaded\PyAudio‑0.2.11‑cp311‑cp311‑win_amd64.whl
     ```

#### ffmpeg Not Found
If you get errors about ffmpeg not being found:
1. Make sure ffmpeg is installed (see Prerequisites section)
2. Verify it's in your PATH by running `ffmpeg -version`
3. If installed but not found, restart your terminal/command prompt
4. On Windows, you might need to restart your computer after adding to PATH

## Usage

### Streamlit Dashboard

**On macOS/Linux:**
```bash
source venv/bin/activate  # Activate virtual environment if not already activated
streamlit run src/dashboard.py
```

**On Windows:**
```cmd
venv\Scripts\activate  # Activate virtual environment if not already activated
streamlit run src/dashboard.py
```

The dashboard will be available at `http://localhost:8501` in your web browser. You can upload audio files and analyze them in real-time.

### FastAPI Endpoint

**On macOS/Linux:**
```bash
source venv/bin/activate  # Activate virtual environment if not already activated
uvicorn src.api:app --reload
```

**On Windows:**
```cmd
venv\Scripts\activate  # Activate virtual environment if not already activated
uvicorn src.api:app --reload
```

The API will be available at `http://localhost:8000` in your web browser. You can:
1. Use the web interface to upload and analyze audio files
2. Send POST requests to `/analyze/` endpoint programmatically
3. Use the batch processing endpoint at `/analyze-batch/` for multiple files
4. View visualizations at `/visualize` after analyzing files

### Jupyter Notebook

**On macOS/Linux:**
```bash
source venv/bin/activate  # Activate virtual environment if not already activated
jupyter notebook notebooks/speech_analysis_demo.ipynb
```

**On Windows:**
```cmd
venv\Scripts\activate  # Activate virtual environment if not already activated
jupyter notebook notebooks/speech_analysis_demo.ipynb
```

Follow the step-by-step demonstration of the pipeline in the notebook.

### Additional API Endpoints

- **`/debug`**: View the current state of analysis results
- **`/reset`**: Clear all analysis results
- **`/visualize`**: View visualizations of analyzed audio files

### Running Both Services Simultaneously

To run both the Streamlit dashboard and FastAPI server simultaneously:

**On macOS/Linux:**
1. Open two terminal windows
2. In the first terminal:
   ```bash
   source venv/bin/activate
   streamlit run src/dashboard.py
   ```
3. In the second terminal:
   ```bash
   source venv/bin/activate
   uvicorn src.api:app --reload
   ```

**On Windows:**
1. Open two Command Prompt windows
2. In the first Command Prompt:
   ```cmd
   venv\Scripts\activate
   streamlit run src/dashboard.py
   ```
3. In the second Command Prompt:
   ```cmd
   venv\Scripts\activate
   uvicorn src.api:app --reload
   ```

## Project Structure

```
speech_analysis/
├── data/
│   └── audio/          # Place audio files here
├── notebooks/
│   └── speech_analysis_demo.ipynb
├── src/
│   ├── audio_processor.py   # Audio preprocessing
│   ├── transcriber.py       # Groq Whisper integration
│   ├── word_recall.py       # Word recall analysis
│   ├── sentence_completion.py # Sentence analysis
│   ├── analyzer.py          # ML pipeline
│   ├── dashboard.py         # Streamlit interface
│   └── api.py              # FastAPI endpoint
├── requirements.txt
└── README.md
```

## API Documentation

### POST /analyze/
Analyzes a single audio file and returns comprehensive speech analysis.

**Parameters:**
- `file`: Audio file (WAV/MP3)
- `baseline_text`: Optional reference text for word recall analysis
- `naming_targets`: Optional comma-separated list of target words for naming task

**Returns JSON with:**
```json
{
  "features": {
    "pause_count": 25,
    "hesitation_count": 0,
    "speech_rate": 97.57,
    "pitch_stats": {
      "pitch_mean": 109.73,
      "pitch_std": 17.36,
      "pitch_range": 122.50
    },
    "transcript": "This morning, I woke up at 7...",
    "word_recall": {
      "similarity_score": 0.85,
      "substitutions": [...],
      "omissions": [...]
    },
    "naming_task": {
      "successful_recalls": [...],
      "failed_attempts": [...],
      "success_rate": 0.8
    },
    "sentence_completion": {
      "complete_sentences": [...],
      "incomplete_sentences": [...],
      "completion_rate": 0.9
    }
  },
  "cognitive_risk_score": 0.87,
  "cognitive_assessment": {
    "risk_level": "Low",
    "indicators": {...}
  },
  "cluster_label": 0,
  "anomaly_score": 0.23,
  "is_anomaly": false,
  "timestamp": "2025-04-18T02:45:12.345678"
}
```

### POST /analyze-batch/
Analyzes multiple audio files and returns comprehensive batch analysis with ML-based anomaly detection.

**Parameters:**
- `files`: List of audio files (WAV/MP3)
- `baseline_text`: Optional reference text for word recall analysis
- `naming_targets`: Optional comma-separated list of target words for naming task

**Returns JSON with:**
```json
{
  "sample_count": 5,
  "anomaly_count": 1,
  "cluster_count": 2,
  "avg_anomaly_score": 0.18,
  "avg_speech_rate": 105.3,
  "avg_pause_count": 22.4,
  "avg_hesitation_count": 3.2,
  "avg_cognitive_risk_score": 0.76,
  "samples": [
    {
      "sample_id": 0,
      "cluster_label": 0,
      "anomaly_score": 0.12,
      "is_anomaly": false,
      "features": {...}
    },
    {
      "sample_id": 1,
      "cluster_label": -1,
      "anomaly_score": 0.87,
      "is_anomaly": true,
      "features": {...}
    },
    ...
  ]
}
```

### GET /visualize
Generates visualizations of analysis results from previously analyzed audio files.

**Parameters:** None

**Returns:** HTML page with interactive visualizations and data tables

## Cognitive Risk Scoring

The system calculates a cognitive risk score (0-1) based on multiple factors:
- Hesitation frequency
- Word recall accuracy
- Naming task performance
- Sentence completion quality

Scores closer to 1.0 indicate better cognitive performance.

## Machine Learning Components

### Unsupervised Anomaly Detection

The system uses unsupervised machine learning to detect anomalies in speech patterns:

1. **Feature Extraction**:
   - Pause count and distribution
   - Hesitation markers frequency
   - Speech rate (words per minute)
   - Pitch variability (standard deviation)
   - Word recall similarity scores
   - Naming task success rates
   - Sentence completion scores

2. **Clustering**:
   - DBSCAN algorithm for density-based clustering
   - Automatically identifies outliers as potential cognitive concerns
   - Groups similar speech patterns together

3. **Anomaly Scoring**:
   - Each sample receives an anomaly score (0-1)
   - Higher scores indicate greater deviation from normal patterns
   - Scores are normalized for easy interpretation

### Batch Analysis

The batch analysis endpoint provides comprehensive statistical analysis across multiple samples:

- Average metrics across all samples
- Identification of outliers and anomalies
- Cluster analysis to group similar speech patterns
- Individual sample details with anomaly scores and cluster labels

## Deliverables and Submission Guidelines

This project fulfills the requirements for the Voice-Based Cognitive Decline Pattern Detection task. Here's how to prepare and submit the deliverables:

### 1. Python Notebook/Script

The project includes a Jupyter notebook that demonstrates the complete pipeline:

```bash
jupyter notebook notebooks/speech_analysis_demo.ipynb
```

To generate results for your report:

1. Place 5-10 audio samples in the `data/audio/` directory
2. Run the notebook cells sequentially
3. Analyze the output and visualizations
4. Save the notebook with your results and analysis

### 2. Sample Visualizations

Visualize feature trends using either:

**A. Streamlit Dashboard:**
```bash
streamlit run src/dashboard.py
```

**B. FastAPI Visualization Endpoint:**
```bash
uvicorn src.api:app --reload
```
Then visit `http://localhost:8000/visualize` after analyzing files.

**C. Jupyter Notebook:**
The notebook includes visualization cells that generate:
- Feature distribution charts
- PCA and t-SNE visualizations for pattern detection
- Anomaly detection results

Capture these visualizations for your report.

### 3. API-Ready Function for Risk Scoring

The project includes a fully functional API with risk scoring capabilities:

```bash
uvicorn src.api:app --reload
```

Test the API using:
1. Web interface at `http://localhost:8000`
2. Direct POST requests to `/analyze/` endpoint
3. Batch processing via `/analyze-batch/` endpoint

The API returns a comprehensive analysis including:
- Cognitive risk score (0-1 scale)
- Anomaly detection results
- Feature extraction metrics
- Cluster analysis

### 4. Final Report

Prepare a report that includes:

1. **Introduction**
   - Project objective and problem statement
   - Dataset description (audio samples used)

2. **Methodology**
   - Feature extraction techniques
   - ML methods used (DBSCAN clustering, anomaly detection)
   - Risk scoring algorithm

3. **Results**
   - Key findings from audio analysis
   - Most insightful features (which speech patterns were most indicative)
   - Visualization of patterns and anomalies
   - Sample risk scores and interpretations

4. **Discussion**
   - Strengths and limitations of the approach
   - Potential clinical applications
   - Next steps for improvement

5. **Conclusion**
   - Summary of findings
   - Recommendations for future work

### Submission Checklist

- [ ] Jupyter notebook with code and results
- [ ] 5-10 audio samples analyzed
- [ ] Visualizations of feature trends
- [ ] API functionality tested and documented
- [ ] Final report with analysis and findings

## Next Steps for Clinical Implementation

1. **Data Collection**:
   - Gather larger dataset of speech samples from diverse populations
   - Include samples from individuals with confirmed cognitive conditions
   - Collect longitudinal data to track changes over time

2. **Model Refinement**:
   - Train supervised models using labeled data
   - Implement more sophisticated feature extraction
   - Optimize anomaly detection thresholds

3. **Clinical Validation**:
   - Conduct validation studies with clinical partners
   - Compare results with established cognitive assessment tools
   - Refine risk scoring based on clinical outcomes

4. **User Experience**:
   - Develop more intuitive visualization tools
   - Create simplified reports for clinical use
   - Implement secure data storage and sharing
