import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from glob import glob

MY_MODEL_NAME = "MGF-LSTM"       
PAPER_MODEL_NAME = "Hybrid"      
GROUND_TRUTH_NAME = "YZ"         

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'lines.linewidth': 2.0
})

COLORS = {
    GROUND_TRUTH_NAME: 'black',
    MY_MODEL_NAME: '#d62728',     
    PAPER_MODEL_NAME: '#ff7f0e',  
    'YZ-LSTM': '#7f7f7f',         
    'GARCH-LSTM': '#1f77b4',      
    'EGARCH-LSTM': '#2ca02c',     
    'TGARCH-LSTM': '#9467bd'      
}

ATTN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def load_logs(log_folder="logs"):
    """
    Scans the logs directory and loads model results into a dictionary with mapped names.
    
    Args:
        log_folder (str): Path to the directory containing JSON log files.
        
    Returns:
        dict: A dictionary where keys are model names and values are the log contents.
    """
    log_files = glob(os.path.join(log_folder, "*.json"))
    data = {}
    
    if not log_files:
        print(f"[!] No .json files found in {log_folder}/")
        return {}

    for filepath in log_files:
        with open(filepath, 'r') as f:
            content = json.load(f)
            
            if 'feature_weights_history' in content:
                model_name = MY_MODEL_NAME
            elif content.get('model_type') == 'Hybrid_Ensemble':
                model_name = PAPER_MODEL_NAME
            elif '-LSTM' in content.get('model_type', ''):
                model_name = content['model_type']
            else:
                model_name = content.get('model_type', 'Unknown_Model')

            data[model_name] = content
            
    return data

def plot_market_overview(ticker="^NSEI", data_folder="data"):
    """
    Generates historical price and log return plots for the primary index.
    
    Args:
        ticker (str): The ticker symbol to plot.
        data_folder (str): Directory where the CSV data is stored.
    """
    clean_name = ticker.replace('^', '')
    file_path = os.path.join(data_folder, f"{clean_name}.csv")
    if not os.path.exists(file_path): 
        print(f"[!] Data file not found: {file_path}")
        return

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if 'Returns' not in df.columns:
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(df.index, df['Close'], color='black', linewidth=1.5)
    ax1.set_ylabel("Index Price")
    ax1.set_title(f"NIFTY 50 Historical Close Price", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(df.index, df['Returns'], color='black', linewidth=1.0)
    ax2.set_ylabel("Log Returns")
    ax2.set_title(f"NIFTY 50 Daily Log Returns", fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("other/NIFTY_50.png", bbox_inches='tight')
    plt.close()
    print("   [+] Saved NIFTY_50.png")

def print_tables(logs, ticker="^NSEI", data_folder="data"):
    """
    Calculates and prints descriptive statistics and model performance tables.
    
    Args:
        logs (dict): Dictionary of loaded model logs.
        ticker (str): The ticker symbol for return calculations.
        data_folder (str): Directory where the CSV data is stored.
    """
    clean_name = ticker.replace('^', '')
    file_path = os.path.join(data_folder, f"{clean_name}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    else:
        print("[!] Data file missing for table generation.")
        return

    try:
        if MY_MODEL_NAME in logs:
            yz_vol = np.array(logs[MY_MODEL_NAME]['predictions']['actual_vol'])
        else:
            first_key = list(logs.keys())[0]
            yz_vol = np.array(logs[first_key]['predictions']['actual_vol'])
    except:
        yz_vol = (returns ** 2).values[-len(returns):]

    stats_data = {
        "Series": ["Returns", GROUND_TRUTH_NAME],
        "Mean": [np.mean(returns), np.mean(yz_vol)],
        "Std Dev": [np.std(returns), np.std(yz_vol)],
        "Skewness": [stats.skew(returns), stats.skew(yz_vol)],
        "Kurtosis": [stats.kurtosis(returns), stats.kurtosis(yz_vol)],
        "J-B Stat": [stats.jarque_bera(returns)[0], stats.jarque_bera(yz_vol)[0]]
    }
    
    lags = [5, 10, 15, 20]
    lb_ret = acorr_ljungbox(returns, lags=lags, return_df=True)
    lb_vol = acorr_ljungbox(yz_vol, lags=lags, return_df=True)
    q_data = {
        "Lag": lags,
        "Ret Q-Stat": lb_ret['lb_stat'].values,
        "Ret p-val": lb_ret['lb_pvalue'].values,
        "Vol Q-Stat": lb_vol['lb_stat'].values,
        "Vol p-val": lb_vol['lb_pvalue'].values
    }

    results = []
    for model_name, data in logs.items():
        m = data['metrics']
        results.append({
            "Model": model_name,
            "RMSE": m['RMSE'],
            "MAE": m['MAE'],
            "MSE": m['MSE'],
            "MAPE (%)": m['MAPE'],
            "VaR 95%": f"{m['Breach_Ratio_95']*100:.2f}%",
            "VaR 99%": f"{m['Breach_Ratio_99']*100:.2f}%"
        })
    df_perf = pd.DataFrame(results).sort_values("MAPE (%)")

    print("\n" + "="*60)
    print("TABLE 1: DESCRIPTIVE STATISTICS")
    print("="*60)
    print(pd.DataFrame(stats_data).round(6).to_string(index=False))
    print("\nTABLE 1 (Part B): Q-STATISTICS")
    print(pd.DataFrame(q_data).round(4).to_string(index=False))
    print("\n" + "="*60)
    print("TABLE 2: MODEL PERFORMANCE")
    print("="*60)
    print(df_perf.to_string(index=False))
    print("="*60 + "\n")

def plot_combined_volatility(logs):
    """
    Visualizes the forecast paths of all models against the ground truth volatility.
    
    Args:
        logs (dict): Dictionary containing predictions for all trained models.
    """
    if MY_MODEL_NAME in logs:
        ref_model = MY_MODEL_NAME
    elif logs:
        ref_model = list(logs.keys())[0]
    else:
        return
    
    dates = pd.to_datetime(logs[ref_model]['predictions']['dates'])
    actual_vol = np.array(logs[ref_model]['predictions']['actual_vol'])
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(dates, actual_vol, color=COLORS[GROUND_TRUTH_NAME], linewidth=2.5, 
             label=GROUND_TRUTH_NAME, zorder=10)
    
    for model_name, content in logs.items():
        if model_name == MY_MODEL_NAME: continue 
        
        preds = np.array(content['predictions']['predicted_vol'])
        min_len = min(len(dates), len(preds))
        col = COLORS.get(model_name, '#999999')
        
        plt.plot(dates[-min_len:], preds[-min_len:], 
                 color=col, linewidth=2.0, alpha=0.7, linestyle='-',
                 label=model_name)

    if MY_MODEL_NAME in logs:
        my_preds = np.array(logs[MY_MODEL_NAME]['predictions']['predicted_vol'])
        plt.plot(dates, my_preds, color=COLORS[MY_MODEL_NAME], linewidth=2.5, 
                 label=MY_MODEL_NAME, zorder=11)

    plt.legend(frameon=True, framealpha=1, fontsize=12, loc='upper left')
    plt.ylabel("Annualized Volatility")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("other/Volatility_Comparison.png", bbox_inches='tight')
    plt.close()
    print("   [+] Saved Volatility_Comparison.png")

def plot_var_analysis(logs):
    """
    Plots multi-level Value-at-Risk (VaR) thresholds against actual daily returns.
    
    Args:
        logs (dict): Logs for the primary MGF-LSTM model.
    """
    if MY_MODEL_NAME not in logs: return
    
    data = logs[MY_MODEL_NAME]['predictions']
    dates = pd.to_datetime(data['dates'])
    returns = np.array(data['Actual_Returns'])
    
    var_90 = np.array(data['VaR_90'])
    var_95 = np.array(data['VaR_95'])
    var_99 = np.array(data['VaR_99'])
    
    plt.figure(figsize=(14, 7))
    
    plt.bar(dates, returns, color='#b0b0b0', alpha=0.6, label='Daily Returns', width=1.0)
    
    plt.plot(dates, var_90, color='#ff7f0e', linestyle=':', linewidth=1.5, label='VaR 90%')
    plt.plot(dates, var_95, color='#d62728', linestyle='--', linewidth=2.0, label='VaR 95%')
    plt.plot(dates, var_99, color='#8c564b', linestyle='-', linewidth=2.0, label='VaR 99%')
    
    breaches_95 = returns < var_95
    plt.scatter(dates[breaches_95], returns[breaches_95], color='black', s=25, zorder=5, label='Breach (95%)')
    
    plt.ylabel("Log Returns")
    plt.xlabel("Date")
    plt.legend(loc='lower left', frameon=True, framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("other/VaR_Analysis.png", bbox_inches='tight')
    plt.close()
    print("   [+] Saved VaR_Analysis.png")

def plot_attention_weights(logs):
    """
    Visualizes time-varying attention weights and average feature importance.
    
    Args:
        logs (dict): Logs for the primary MGF-LSTM model.
    """
    if MY_MODEL_NAME not in logs: return
    log = logs[MY_MODEL_NAME]
    
    if 'feature_weights_history' not in log: return

    weights = np.array(log['feature_weights_history'])
    features = log.get('feature_names', ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
    dates = pd.to_datetime(log['predictions']['dates'])
    
    plt.figure(figsize=(14, 6))
    for i, feature in enumerate(features):
        col = ATTN_COLORS[i % len(ATTN_COLORS)]
        plt.plot(dates, weights[:, i], label=feature, linewidth=2.0, color=col)
        
    plt.ylabel("Attention Weight")
    plt.xlabel("Date")
    plt.legend(loc='upper left', frameon=True, framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05)
    plt.savefig("other/Attention_Weights_Time.png", bbox_inches='tight')
    plt.close()
    print("   [+] Saved Attention_Weights_Time.png")
    
    avg_weights = weights.mean(axis=0)
    print("Average Importance Weights are: ")
    print(avg_weights, "\n")
    df_bar = pd.DataFrame({'Feature': features, 'Weight': avg_weights})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_bar, x='Feature', y='Weight', palette=ATTN_COLORS)
    
    plt.ylabel("Average Weight")
    plt.xlabel("Input Feature")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("other/Average_Feature_Importance.png", bbox_inches='tight')
    plt.close()
    print("   [+] Saved Average_Feature_Importance.png")

def main():
    """
    Main entry point for generating all visual components of the research paper.
    """
    print("Generating Visualizations...\n")
    logs = load_logs()
    if not logs: return

    plot_market_overview()
    print_tables(logs)
    plot_combined_volatility(logs)
    plot_var_analysis(logs)
    plot_attention_weights(logs)

    print("\nDone. All figures generated in other/ folder.")

if __name__ == "__main__":
    main()