# main.py
import argparse
import sys
import logging
from config import Config
from data_loader import load_data, prepare_data
from model import SentimentModel

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_mode(show_plot: bool):
    """
    Executes the training pipeline.
    
    Args:
        show_plot (bool): If True, displays the cost history plot after training.
    """
    logger.info("--- Starting Training Pipeline ---")
    
    # Load and Prepare Data
    try:
        pos, neg = load_data()
        train_x, test_x, train_y, test_y = prepare_data(pos, neg)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Initialize and Train Model
    model = SentimentModel()
    model.fit(train_x, train_y, Config.LEARNING_RATE, Config.NUM_ITERATIONS)
    
    # Evaluate
    accuracy = model.evaluate(test_x, test_y)
    logger.info(f"Model Accuracy on Test Set: {accuracy:.2%}")
    
    # Save Model
    model.save(Config.MODEL_PATH)

    # Visualize
    if show_plot:
        logger.info("Visualizing training history...")
        model.plot_history()

def inference_mode(tweet: str):
    """
    Executes the inference pipeline for a single tweet.
    
    Args:
        tweet (str): The input text to classify.
    """
    model = SentimentModel()
    try:
        model.load(Config.MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model file not found. Please train the model first using --mode train")
        return

    score = model.predict_proba(tweet)
    label = "POSITIVE" if score > 0.5 else "NEGATIVE"
    
    print(f"\nTweet: {tweet}")
    print(f"Sentiment: {label}")  # (Score: {score:.4f})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis System")
    
    # Required argument: mode
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, 
                        help="Execution mode: 'train' to build model, 'predict' to classify text")
    
    # Optional argument for prediction
    parser.add_argument('--text', type=str, help="Text string to predict (required for predict mode)")
    
    # Optional argument for training visualization
    parser.add_argument('--plot', action='store_true', help="Show training cost plot (only for train mode)")

    args = parser.parse_args()

    if args.mode == 'train':
        train_mode(show_plot=args.plot)
        
    elif args.mode == 'predict':
        if not args.text:
            logger.error("Argument --text is required for predict mode.")
            print("Usage example: python main.py --mode predict --text \"I love this!\"")
        else:
            inference_mode(args.text)