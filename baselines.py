import os
import json
import numpy as np
import pandas as pd
from arch import arch_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from dataset import Dataset

class VolatilityCalculators:
    @staticmethod
    def calculate_yang_zhang_volatility(df, window=22):
        """
        Calculates the drift-independent Yang-Zhang volatility proxy (ground truth).
        
        Args:
            df (pd.DataFrame): Dataframe containing OHLC data.
            window (int): Rolling window for variance calculation.
            
        Returns:
            pd.Series: Annualized Yang-Zhang volatility.
        """
        epsilon = 1e-8
        prev_close = df['Close'].shift(1)
        o = np.log((df['Open'] / prev_close).replace(0, epsilon))
        c = np.log((df['Close'] / df['Open']).replace(0, epsilon))
        h = np.log((df['High'] / df['Open']).replace(0, epsilon))
        l = np.log((df['Low'] / df['Open']).replace(0, epsilon))
        
        sigma_o_var = o.rolling(window).var()
        sigma_c_var = c.rolling(window).var()
        rs_var = (h * (h - c)) + (l * (l - c))
        sigma_rs_var = rs_var.rolling(window).mean()
        
        # Weighted combination of overnight and intraday volatility
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_variance = sigma_o_var + (k * sigma_c_var) + ((1 - k) * sigma_rs_var)
        return np.sqrt(yz_variance) * np.sqrt(252)

    @staticmethod
    def get_garch_volatility(returns, model_type='GARCH'):
        """
        Fits baseline statistical models to generate volatility features.
        
        Args:
            returns (pd.Series): Time series of log returns.
            model_type (str): Variant of GARCH to fit ('GARCH', 'EGARCH', 'TGARCH').
            
        Returns:
            tuple: (Annualized volatility series, model parameters).
        """
        scaled_returns = returns * 100 
        try:
            if model_type == 'GARCH':
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
            elif model_type == 'EGARCH':
                model = arch_model(scaled_returns, vol='EGARCH', p=1, q=1, rescale=False)
            elif model_type == 'TGARCH':
                model = arch_model(scaled_returns, vol='GARCH', p=1, o=1, q=1, rescale=False)
            res = model.fit(disp='off', show_warning=False)
            
            daily_vol_decimal = res.conditional_volatility / 100
            return daily_vol_decimal * np.sqrt(252), res.params.to_dict()
        except:
            return pd.Series(np.zeros(len(returns)), index=returns.index), {}

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initializes a vanilla LSTM architecture.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): LSTM hidden layer size.
        """
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        Forward pass for the baseline LSTM model.
        
        Args:
            x (torch.Tensor): Input sequence tensor.
            
        Returns:
            torch.Tensor: Predicted volatility (positive-constrained).
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        raw_out = self.fc(last_hidden)
        # Softplus ensures non-negative volatility for numerical stability
        prediction = F.softplus(raw_out) 
        return prediction

class BaselinePredictor:
    def __init__(self, ticker, df_ticker, input_type, seq_len=10):
        """
        Manages training and evaluation for single-input baseline models.
        
        Args:
            ticker (str): Ticker name.
            df_ticker (pd.DataFrame): Raw price data.
            input_type (str): The specific volatility feature used as input.
            seq_len (int): Sliding window size.
        """
        self.ticker = ticker
        self.df = df_ticker.copy()
        self.input_type = input_type
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.model = None
        self.logs = {
            "model_type": f"{input_type}-LSTM",
            "metrics": {},
            "predictions": {}
        }

    def prepare_data(self):
        """
        Engineers features based on the specified baseline input type.
        
        Returns:
            pd.DataFrame: Processed dataframe for baseline training.
        """
        self.df['Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['YZ_Vol'] = VolatilityCalculators.calculate_yang_zhang_volatility(self.df)
        self.df.dropna(inplace=True)
        
        if self.input_type == 'YZ':
            self.df['Feature_Vol'] = self.df['YZ_Vol']
        else:
            vol_est, _ = VolatilityCalculators.get_garch_volatility(self.df['Returns'], self.input_type)
            self.df['Feature_Vol'] = vol_est
            
        self.df.dropna(inplace=True)
        return self.df

    def qlike_loss(self, y_pred, y_true):
        """
        Calculates Quasi-Likelihood loss to penalize under-prediction of risk.
        
        Args:
            y_pred (torch.Tensor): Model predictions.
            y_true (torch.Tensor): Target (YZ) volatility.
            
        Returns:
            torch.Tensor: Mean QLIKE loss.
        """
        y_pred = y_pred + 1e-6 
        loss = 2 * torch.log(y_pred) + (y_true**2 / y_pred**2)
        return torch.mean(loss)

    def train(self, epochs=500):
        """
        Trains the simple LSTM on a single statistical feature baseline.
        
        Args:
            epochs (int): Training iterations.
            
        Returns:
            tuple: (Predictions, Actuals, Returns, Dates).
        """
        feature_cols = ['Feature_Vol', 'Returns']
        target_col = 'YZ_Vol'
        
        data_vals = self.df[feature_cols].values
        target_vals = self.df[target_col].values
        
        split = int(0.8 * len(data_vals))
        train_X_raw, test_X_raw = data_vals[:split], data_vals[split:]
        train_y_raw, test_y_raw = target_vals[:split], target_vals[split:]
        
        self.scaler.fit(train_X_raw)
        scaled_train = self.scaler.transform(train_X_raw)
        scaled_test = self.scaler.transform(test_X_raw)
        
        def create_sequences(data, target):
            X, y = [], []
            for i in range(len(data) - self.seq_len):
                X.append(data[i : i + self.seq_len])
                y.append(target[i + self.seq_len])
            return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(1)

        X_train, y_train = create_sequences(scaled_train, train_y_raw)
        X_test, y_test = create_sequences(scaled_test, test_y_raw)
        
        self.model = SimpleLSTM(input_dim=len(feature_cols))
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        print(f"   Training {self.input_type}-LSTM...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X_train)
            loss = self.qlike_loss(preds, y_train)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            final_preds = self.model(X_test).numpy().flatten()
            actuals = y_test.numpy().flatten()
            
            mse = np.mean((final_preds - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(final_preds - actuals))
            safe_actuals = np.where(actuals == 0, 1e-6, actuals)
            mape = np.mean(np.abs((actuals - final_preds) / safe_actuals)) * 100

            pred_daily_vol = final_preds / np.sqrt(252)
            actual_daily_returns = self.df['Returns'].values[split + self.seq_len:]
            
            var_90 = -1.282 * pred_daily_vol
            var_95 = -1.645 * pred_daily_vol
            var_99 = -2.326 * pred_daily_vol
            
            br_90 = np.sum(actual_daily_returns < var_90) / len(actual_daily_returns)
            br_95 = np.sum(actual_daily_returns < var_95) / len(actual_daily_returns)
            br_99 = np.sum(actual_daily_returns < var_99) / len(actual_daily_returns)
            
            self.logs['predictions'] = {
                "dates": self.df.index[split + self.seq_len:].strftime('%Y-%m-%d').tolist(),
                "actual_vol": actuals.tolist(),
                "predicted_vol": final_preds.tolist(),
                "VaR_90": var_90.tolist(),
                "VaR_95": var_95.tolist(),
                "VaR_99": var_99.tolist(),
                "Actual_Returns": actual_daily_returns.tolist()
            }
            
            self.logs['metrics'] = {
                "RMSE": float(rmse),
                "MAE": float(mae),
                "MSE": float(mse),
                "MAPE": float(mape),
                "Breach_Ratio_90": float(br_90),
                "Breach_Ratio_95": float(br_95),
                "Breach_Ratio_99": float(br_99)
            }
            
            print(f"   [{self.input_type}-LSTM] RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
            return final_preds, actuals, actual_daily_returns, self.logs['predictions']['dates']

def save_logs(model_type, logs, folder="logs"):
    """
    Saves baseline metrics to the local logs directory.
    
    Args:
        model_type (str): Baseline model identifier.
        logs (dict): Results and metrics.
        folder (str): Target folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"{model_type}_metrics.json"
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"   Saved log: {filename}")

def main():
    """
    Executes the baseline comparison suite (YZ, GARCH, EGARCH, TGARCH) and the static hybrid ensemble.
    """
    print("Initializing Dataset for Baselines...")
    ds = Dataset()
    master_dict = ds.get_price_data()
    nifty_df = master_dict['NIFTY50']
    
    baselines = ['YZ', 'GARCH', 'EGARCH', 'TGARCH']
    model_preds = {}
    actuals, returns, dates = None, None, None
    
    print(f"\nSTARTING BASELINE COMPARISON")
    print("="*50)
    
    for model_type in baselines:
        print(f"\n--- Running Model: {model_type}-LSTM ---")
        try:
            predictor = BaselinePredictor("NIFTY_50", nifty_df, model_type)
            predictor.prepare_data()
            preds, acts, rets, dts = predictor.train(epochs=500)
            
            model_preds[model_type] = preds
            actuals, returns, dates = acts, rets, dts
            
            save_logs(f"{model_type}_LSTM", predictor.logs)
        except Exception as e:
            print(f"FAILED {model_type}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n--- Calculating Hybrid Ensemble (Static Average) ---")
    if len(model_preds) == 4:
        # Computes time-invariant average of all baseline model predictions
        ensemble_preds = np.mean(list(model_preds.values()), axis=0)
        
        mse = np.mean((ensemble_preds - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_preds - actuals))
        mape = np.mean(np.abs((actuals - ensemble_preds) / np.where(actuals == 0, 1e-6, actuals))) * 100
        
        pred_daily_vol = ensemble_preds / np.sqrt(252)
        v90 = -1.282 * pred_daily_vol
        v95 = -1.645 * pred_daily_vol
        v99 = -2.326 * pred_daily_vol
        
        br90 = np.sum(returns < v90) / len(returns)
        br95 = np.sum(returns < v95) / len(returns)
        br99 = np.sum(returns < v99) / len(returns)
        
        ensemble_logs = {
            "model_type": "Hybrid_Ensemble",
            "metrics": {
                "RMSE": float(rmse), 
                "MAE": float(mae),
                "MAPE": float(mape), 
                "MSE": float(mse),
                "Breach_Ratio_90": float(br90),
                "Breach_Ratio_95": float(br95), 
                "Breach_Ratio_99": float(br99)
            },
            "predictions": {
                "dates": dates, 
                "actual_vol": actuals.tolist(), 
                "predicted_vol": ensemble_preds.tolist(),
                "VaR_90": v90.tolist(), 
                "VaR_95": v95.tolist(), 
                "VaR_99": v99.tolist(), 
                "Actual_Returns": returns.tolist()
            }
        }
        print(f"   [Hybrid] RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
        save_logs("Hybrid", ensemble_logs)
    
    print("\nALL MODELS COMPLETED.")

if __name__ == "__main__":
    main()