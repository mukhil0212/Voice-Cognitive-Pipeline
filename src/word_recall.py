import difflib
from typing import Dict, List, Tuple
import re

class WordRecallDetector:
    def __init__(self):
        self.sequence_matcher = difflib.SequenceMatcher(None)
    
    def analyze_recall(self, transcript: str, prompt: str) -> Dict:
        """
        Analyze word recall by comparing transcript against prompt.
        Returns substitutions, omissions, and overall similarity score.
        """
        # Normalize text
        transcript_words = re.findall(r'\b\w+\b', transcript.lower())
        prompt_words = re.findall(r'\b\w+\b', prompt.lower())
        
        self.sequence_matcher.set_seqs(prompt_words, transcript_words)
        
        # Find substitutions and omissions
        substitutions = []
        omissions = []
        
        for tag, i1, i2, j1, j2 in self.sequence_matcher.get_opcodes():
            if tag == 'replace':
                # Word substitution
                substitutions.append({
                    'expected': ' '.join(prompt_words[i1:i2]),
                    'actual': ' '.join(transcript_words[j1:j2])
                })
            elif tag == 'delete':
                # Word omission
                omissions.append(' '.join(prompt_words[i1:i2]))
        
        return {
            'similarity_score': self.sequence_matcher.ratio(),
            'substitutions': substitutions,
            'omissions': omissions,
            'substitution_count': len(substitutions),
            'omission_count': len(omissions)
        }

class NamingTaskAnalyzer:
    def __init__(self, target_words: List[str]):
        """
        Initialize with a list of target words for the naming task.
        """
        self.target_words = [word.lower() for word in target_words]
        self.sequence_matcher = difflib.SequenceMatcher(None)
    
    def find_closest_match(self, word: str) -> Tuple[str, float]:
        """Find the closest matching target word."""
        word = word.lower()
        best_ratio = 0
        best_match = None
        
        for target in self.target_words:
            self.sequence_matcher.set_seqs(target, word)
            ratio = self.sequence_matcher.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = target
        
        return best_match, best_ratio
    
    def analyze_naming(self, transcript: str) -> Dict:
        """
        Analyze naming task performance.
        Returns successful recalls and failed attempts.
        """
        words = re.findall(r'\b\w+\b', transcript.lower())
        
        successful_recalls = []
        failed_attempts = []
        
        for word in words:
            closest_match, similarity = self.find_closest_match(word)
            if similarity > 0.8:  # Threshold for successful recall
                successful_recalls.append({
                    'target': closest_match,
                    'actual': word,
                    'similarity': similarity
                })
            elif similarity > 0.5:  # Threshold for failed attempt
                failed_attempts.append({
                    'target': closest_match,
                    'actual': word,
                    'similarity': similarity
                })
        
        return {
            'successful_recalls': successful_recalls,
            'failed_attempts': failed_attempts,
            'success_rate': len(successful_recalls) / len(self.target_words) if self.target_words else 0
        }
