import librosa
import numpy as np
from pydub import AudioSegment
import parselmouth
from typing import Dict, List, Tuple

class AudioProcessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load_and_preprocess(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        # Load audio
        y, sr = librosa.load(file_path, sr=self.target_sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize volume
        y = librosa.util.normalize(y)
        
        return y, sr
    
    def extract_pauses(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Extract silence intervals."""
        intervals = librosa.effects.split(y, top_db=20)
        pauses = []
        for i in range(len(intervals)-1):
            pause_start = intervals[i][1] / sr
            pause_end = intervals[i+1][0] / sr
            pauses.append((pause_start, pause_end))
        return pauses
    
    def compute_pitch_stats(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute pitch statistics using Parselmouth."""
        sound = parselmouth.Sound(y, sr)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        
        # Remove zero values (unvoiced)
        pitch_values = pitch_values[pitch_values != 0]
        
        return {
            'pitch_mean': float(np.mean(pitch_values)),
            'pitch_std': float(np.std(pitch_values)),
            'pitch_range': float(np.ptp(pitch_values))
        }
    
    def compute_speech_rate(self, word_count: int, duration: float) -> float:
        """Compute speech rate in words per minute."""
        return (word_count / duration) * 60
