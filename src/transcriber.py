from groq import Groq
import re
from typing import Dict, Optional
from word_recall import WordRecallDetector, NamingTaskAnalyzer
from sentence_completion import SentenceCompletionDetector

class Transcriber:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.word_recall = WordRecallDetector()
        self.sentence_completion = SentenceCompletionDetector()
        self.naming_analyzer = None  # Initialize later with target words

    def set_naming_targets(self, target_words: list):
        """Set target words for naming task analysis."""
        self.naming_analyzer = NamingTaskAnalyzer(target_words)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Groq's Whisper model."""
        try:
            # Note: This is a placeholder for Groq's Whisper API call
            # Actual implementation will depend on Groq's API structure
            response = self.client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=open(audio_path, "rb")
            )
            return response.text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def analyze_transcript(self, transcript: str, baseline: Optional[str] = None) -> Dict:
        """Comprehensive analysis of transcript including linguistic markers,
        word recall, naming tasks, and sentence completion."""
        results = {}

        # Basic linguistic analysis
        hesitations = len(re.findall(r'\b(um|uh|er)\b', transcript.lower()))
        words = len(transcript.split())

        results.update({
            'hesitation_count': hesitations,
            'word_count': words,
            'hesitation_rate': hesitations / words if words > 0 else 0
        })

        # Sentence completion analysis
        completion_analysis = self.sentence_completion.analyze_completion(transcript)
        results['sentence_completion'] = completion_analysis
        results['completion_score'] = self.sentence_completion.get_completion_score(transcript)

        # Word recall analysis (if baseline provided)
        if baseline:
            recall_analysis = self.word_recall.analyze_recall(transcript, baseline)
            results['word_recall'] = recall_analysis

        # Naming task analysis (if targets set)
        if self.naming_analyzer:
            naming_analysis = self.naming_analyzer.analyze_naming(transcript)
            results['naming_task'] = naming_analysis

        return results

    def get_cognitive_risk_score(self, analysis: Dict) -> float:
        """Calculate an overall cognitive risk score based on all analyses."""
        scores = [
            1 - min(analysis.get('hesitation_rate', 0) * 5, 1),  # Penalize high hesitation rates
            analysis.get('completion_score', 1),  # Sentence completion score
            analysis.get('word_recall', {}).get('similarity_score', 1),  # Word recall score
            analysis.get('naming_task', {}).get('success_rate', 1)  # Naming task score
        ]

        # Remove None values
        scores = [s for s in scores if s is not None]

        # Return average score if we have any valid scores
        return sum(scores) / len(scores) if scores else 0.0
