# model.py
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from utils import sigmoid, cost_function
from features import extract_features
from preprocess import build_freqs
from config import Config

# Setup Logger
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentModel:
    """Logistic Regression Model for Sentiment Analysis."""

    def __init__(self):
        self.theta: Optional[np.ndarray] = None
        self.freqs: Dict[Tuple[str, int], int] = {}
        self.costs: List[float] = []

    def fit(self, train_x_raw: List[str], train_y: np.ndarray, 
            learning_rate: float, num_iters: int, verbose: bool = True) -> None:
        """Trains the model using Gradient Descent.

        Args:
            train_x_raw (List[str]): List of raw tweet strings.
            train_y (np.ndarray): Target labels (m, 1).
            learning_rate (float): Step size for gradient descent.
            num_iters (int): Number of iterations.
            verbose (bool): Whether to log progress.
        """
        logger.info("Building frequency dictionary...")
        self.freqs = build_freqs(train_x_raw, train_y)
        
        logger.info("Extracting features...")
        m = len(train_x_raw)
        X = np.zeros((m, Config.NUM_FEATURES))
        
        for i in range(m):
            # Using Config for consistency, assumption: extract_features handles padding
            X[i, :] = extract_features(train_x_raw[i], self.freqs)
        
        # Init theta
        self.theta = np.zeros((Config.NUM_FEATURES, 1))

        logger.info(f"Training for {num_iters} iterations...")
        for i in range(num_iters):
            z = np.dot(X, self.theta)
            y_hat = sigmoid(z)
            cost = cost_function(y_hat, train_y)
            self.costs.append(cost)
            
            gradient = (1 / m) * np.dot(X.T, (y_hat - train_y))
            self.theta = self.theta - learning_rate * gradient
            
            if verbose and i % 100 == 0:
                logger.info(f"Iter {i}: Cost {cost:.6f}")
        
        logger.info("Training completed.")

    def predict_proba(self, tweet: str) -> float:
        """Predicts probability of positive sentiment.

        Args:
            tweet (str): Input text.

        Returns:
            float: Probability (0.0 to 1.0).
        """
        if self.theta is None:
            logger.error("Model not trained yet.")
            raise RuntimeError("Model parameters (theta) are missing. Call fit() or load().")
            
        x_features = extract_features(tweet, self.freqs)
        y_pred = sigmoid(np.dot(x_features, self.theta))
        return float(y_pred.item())
    
    def predict(self, tweet, threshold=0.5):
        """Return label (0 or 1)"""
        return 1 if self.predict_proba(tweet) > threshold else 0

    def evaluate(self, test_x, test_y):
        """Evaluate accuracy"""
        correct = 0
        for i, tweet in enumerate(test_x):
            pred = self.predict(tweet)
            if pred == test_y[i]:
                correct += 1
        return correct / len(test_x)

    def save(self, filepath: str) -> None:
        """Saves model parameters to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({'theta': self.theta, 'freqs': self.freqs}, f)
            logger.info(f"Model saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save model: {e}")

    def load(self, filepath: str) -> None:
        """Loads model parameters from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.theta = data['theta']
                self.freqs = data['freqs']
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {filepath}")
            raise

    def plot_history(self) -> None:
        """Visualizes the cost reduction over iterations."""
        if not self.costs:
            logger.warning("No training history to plot.")
            return
            
        plt.plot(self.costs)
        plt.ylabel('Cost J')
        plt.xlabel('Iterations')
        plt.title('Training History')
        plt.show()