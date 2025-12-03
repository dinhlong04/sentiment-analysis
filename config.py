# config.py
import os
import logging

class Config:
    """Configuration class for hyperparameters and paths."""
    
    # Hyperparameters
    LEARNING_RATE: float = 1e-9
    NUM_ITERATIONS: int = 1500
    TEST_SPLIT_INDEX: int = 4000
    NUM_FEATURES: int = 3  # Moved magic number here
    
    # Paths & System
    MODEL_PATH: str = "saved_model.pkl"
    NLTK_DATA: list = ['twitter_samples', 'stopwords']
    LOG_LEVEL = logging.INFO