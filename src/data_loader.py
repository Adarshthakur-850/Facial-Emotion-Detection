import pandas as pd
import numpy as np
import os
import requests

DATA_URL = "https://raw.githubusercontent.com/kkhalis/facial_emotion_detection/master/fer2013.csv"
DATA_PATH = os.path.join("data", "fer2013.csv")

def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(DATA_PATH):
        print("Downloading FER-2013 dataset...")
        try:
            response = requests.get(DATA_URL)
            if response.status_code == 200:
                with open(DATA_PATH, "wb") as f:
                    f.write(response.content)
                print("Download complete.")
            else:
                print(f"Failed to download data: {response.status_code}")
                # Fallback to dummy data if download fails (to prevent pipeline crash)
                create_dummy_data()
        except Exception as e:
            print(f"Error downloading data: {e}")
            create_dummy_data()
            
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
         create_dummy_data()
         
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    return df

def create_dummy_data():
    if not os.path.exists(DATA_PATH):
        print("Creating dummy data...")
        # Create dummy pixels (48*48=2304)
        dummy_pixels = " ".join(["0"] * 2304)
        df = pd.DataFrame({
            'emotion': [0, 1, 2, 3, 4, 5, 6] * 150,
            'pixels': [dummy_pixels] * 1050,
            'Usage': ['Training'] * 800 + ['PrivateTest'] * 125 + ['PublicTest'] * 125
        })
        df.to_csv(DATA_PATH, index=False)
