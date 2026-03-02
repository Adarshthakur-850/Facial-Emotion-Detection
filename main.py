import sys
import os
import argparse
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import run_real_time_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or live')
    args = parser.parse_args()
    
    if args.mode == 'live':
        run_real_time_inference()
        return

    print("Starting Training Pipeline...")
    
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    model = build_model()
    
    train_model(model, X_train, y_train, X_val, y_val, epochs=5) # 5 epochs for demo, use more for real results
    
    evaluate_model(model, X_test, y_test)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline Failed: {e}")
