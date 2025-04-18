import spacy
from typing import Dict, List
import re

class SentenceCompletionDetector:
    def __init__(self):
        """Initialize spaCy model for dependency parsing."""
        self.nlp = spacy.load("en_core_web_sm")
    
    def _has_main_verb(self, doc) -> bool:
        """Check if the sentence has a main verb."""
        return any(token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"] for token in doc)
    
    def _has_subject(self, doc) -> bool:
        """Check if the sentence has a subject."""
        return any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc)
    
    def _is_fragment(self, doc) -> bool:
        """
        Detect if a sentence is a fragment based on:
        - Missing main verb
        - Missing subject
        - Incomplete clausal structure
        """
        if not self._has_main_verb(doc):
            return True
        if not self._has_subject(doc):
            return True
        
        # Check for incomplete dependent clauses
        has_subordinate_marker = any(token.dep_ == "mark" for token in doc)
        if has_subordinate_marker and not any(token.dep_ == "ROOT" for token in doc):
            return True
        
        return False
    
    def _get_fragment_type(self, doc) -> str:
        """Identify the type of sentence fragment."""
        if not self._has_main_verb(doc):
            return "missing_verb"
        if not self._has_subject(doc):
            return "missing_subject"
        return "incomplete_clause"
    
    def analyze_completion(self, text: str) -> Dict:
        """
        Analyze sentence completion in the given text.
        Returns statistics about complete vs incomplete sentences.
        """
        # Split into sentences (handling both punctuation and verbal pauses)
        sentences = re.split(r'[.!?]+|(?<=[a-z])\s+(?:um|uh|er)\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        results = {
            'complete_sentences': [],
            'incomplete_sentences': [],
            'completion_rate': 0.0,
            'fragment_types': {
                'missing_verb': 0,
                'missing_subject': 0,
                'incomplete_clause': 0
            }
        }
        
        for sentence in sentences:
            doc = self.nlp(sentence)
            
            if self._is_fragment(doc):
                fragment_type = self._get_fragment_type(doc)
                results['incomplete_sentences'].append({
                    'text': sentence,
                    'type': fragment_type
                })
                results['fragment_types'][fragment_type] += 1
            else:
                results['complete_sentences'].append(sentence)
        
        total_sentences = len(sentences)
        if total_sentences > 0:
            results['completion_rate'] = len(results['complete_sentences']) / total_sentences
        
        return results
    
    def get_completion_score(self, text: str) -> float:
        """
        Get a normalized score (0-1) for sentence completion quality.
        Takes into account both the completion rate and the types of fragments.
        """
        analysis = self.analyze_completion(text)
        
        # Base score is the completion rate
        score = analysis['completion_rate']
        
        # Penalize based on fragment types (missing subject/verb is worse than incomplete clause)
        total_fragments = sum(analysis['fragment_types'].values())
        if total_fragments > 0:
            weighted_penalty = (
                analysis['fragment_types']['missing_verb'] * 0.4 +
                analysis['fragment_types']['missing_subject'] * 0.4 +
                analysis['fragment_types']['incomplete_clause'] * 0.2
            ) / total_fragments
            
            score -= weighted_penalty * 0.5  # Scale penalty impact
        
        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
