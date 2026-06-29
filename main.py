import os
import json
import pandas as pd
from dataset import Dataset
from models import HybridPredictor

def save_logs(model_name, logs, folder="logs"):
    """
    Saves the experiment results and evaluation metrics to a JSON file.
    
    Args:
        model_name (str): Name of the model configuration (e.g., 'MGF-LSTM').
        logs (dict): Dictionary containing training logs and performance metrics.
        folder (str): Target directory for saving log files.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, f"{model_name}_metrics.json")
    
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"   [+] Logs saved to {filepath}")

def main():
    """
    Orchestrates the full pipeline: data loading, feature engineering, 
    model training, and results logging.
    """
    print("Initializing Dataset...")
    ds = Dataset()
    
    # Retrieves aligned OHLC and India VIX data
    master_dict = ds.get_price_data()
    nifty_df = master_dict['NIFTY50']
    vix_df = master_dict['VIX']
    
    model_name = "MGF-LSTM"
    
    print(f"\n" + "="*60)
    print(f"STARTING EXPERIMENT ON: {model_name}")
    print(f"Data Points: {len(nifty_df)} days")
    print("="*60)
    
    try:
        # Initialize the Hybrid Attention-LSTM predictor
        predictor = HybridPredictor(model_name, nifty_df, vix_df)
        
        # Feature extraction includes GARCH family models and YZ proxies
        print("   Step 1: Preparing Feature Set (GARCH/EGARCH/TGARCH/YZ)...")
        predictor.prepare_data()
        
        # Train model using QLIKE loss and dynamic attention
        print("   Step 2: Training Hybrid Attention-LSTM...")
        predictor.train(epochs=500)
        
        # Capture RMSE, MAPE, and Value-at-Risk (VaR) breach ratios
        print("   Step 3: Saving Experiment Logs...")
        save_logs(model_name, predictor.logs)
        
        print("\n" + "="*60)
        print(f"SUCCESS: Experiment completed for {model_name}")
        print("="*60)
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE processing {model_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()