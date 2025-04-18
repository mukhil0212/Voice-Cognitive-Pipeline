import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional

class SpeechAnalyzer:
    def __init__(self, use_dbscan: bool = True):
        self.use_dbscan = use_dbscan
        self.scaler = StandardScaler()
        if use_dbscan:
            self.model = DBSCAN(eps=0.5, min_samples=2)
        else:
            self.model = IsolationForest(contamination=0.1, random_state=42)

    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to a matrix."""
        feature_matrix = []
        for features in features_list:
            # Extract word recall similarity if available
            word_recall_similarity = 0
            if 'word_recall' in features and features['word_recall']:
                word_recall_similarity = features['word_recall'].get('similarity_score', 0)

            # Extract naming task success rate if available
            naming_success_rate = 0
            if 'naming_task' in features and features['naming_task']:
                naming_success_rate = features['naming_task'].get('success_rate', 0)

            # Extract sentence completion score if available
            completion_score = features.get('completion_score', 0)

            feature_vector = [
                features['pause_count'],
                features['hesitation_count'],
                features['speech_rate'],
                features['pitch_stats']['pitch_std'],
                word_recall_similarity,
                naming_success_rate,
                completion_score
            ]
            feature_matrix.append(feature_vector)
        return np.array(feature_matrix)

    def fit_predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features and perform anomaly detection.

        Returns:
            Tuple containing (cluster_labels, anomaly_scores)
        """
        scaled_features = self.scaler.fit_transform(features)

        if self.use_dbscan:
            labels = self.model.fit_predict(scaled_features)
            # Calculate distance to nearest cluster center as anomaly score
            anomaly_scores = np.zeros(len(features))
            for i, label in enumerate(labels):
                if label == -1:  # Outlier
                    anomaly_scores[i] = 1.0
                else:
                    # Find distance to cluster center
                    cluster_points = scaled_features[labels == label]
                    cluster_center = np.mean(cluster_points, axis=0)
                    distance = np.linalg.norm(scaled_features[i] - cluster_center)
                    # Normalize to 0-1 range (higher = more anomalous)
                    anomaly_scores[i] = min(distance / 5.0, 1.0)  # Cap at 1.0
        else:
            raw_scores = self.model.score_samples(scaled_features)
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = 1 - (raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-10)
            # Get labels (1: normal, -1: anomaly)
            labels = self.model.predict(scaled_features)
            # Convert IsolationForest labels (1: normal, -1: anomaly) to match DBSCAN
            labels = np.where(labels == 1, 0, -1)

        return labels, anomaly_scores

    def analyze_batch(self, features_list: List[Dict]) -> Dict:
        """Analyze a batch of speech samples and return summary statistics.

        Args:
            features_list: List of feature dictionaries from multiple speech samples

        Returns:
            Dictionary with summary statistics and anomaly detection results
        """
        if not features_list:
            return {"error": "No features provided for analysis"}

        # Prepare feature matrix
        feature_matrix = self.prepare_features(features_list)

        # Perform anomaly detection
        labels, anomaly_scores = self.fit_predict(feature_matrix)

        # Calculate summary statistics
        summary = {
            "sample_count": len(features_list),
            "anomaly_count": np.sum(labels == -1),
            "cluster_count": len(set(labels)) - (1 if -1 in labels else 0),
            "avg_anomaly_score": float(np.mean(anomaly_scores)),
            "avg_speech_rate": float(np.mean([f["speech_rate"] for f in features_list])),
            "avg_pause_count": float(np.mean([f["pause_count"] for f in features_list])),
            "avg_hesitation_count": float(np.mean([f["hesitation_count"] for f in features_list])),
            "avg_cognitive_risk_score": float(np.mean([f.get("cognitive_risk_score", 0) for f in features_list])),
            "samples": []
        }

        # Add individual sample results with cluster labels and anomaly scores
        for i, (features, label, score) in enumerate(zip(features_list, labels, anomaly_scores)):
            sample_result = {
                "sample_id": i,
                "cluster_label": int(label),
                "anomaly_score": float(score),
                "is_anomaly": bool(label == -1),
                "features": features
            }
            summary["samples"].append(sample_result)

        return summary

    def visualize_results(self, features: np.ndarray, labels: np.ndarray) -> Tuple[plt.Figure, plt.Figure]:
        """Create visualization plots."""
        scaled_features = self.scaler.transform(features)

        # PCA plot
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
        ax1.set_title('PCA Visualization of Speech Features')
        plt.colorbar(scatter1)

        # t-SNE plot
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(scaled_features)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
        ax2.set_title('t-SNE Visualization of Speech Features')
        plt.colorbar(scatter2)

        return fig1, fig2
