import numpy as np
import pandas as pd
from arch import arch_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class VolatilityCalculators:
    @staticmethod
    def calculate_yang_zhang_volatility(df, window=22):
        """
        Calculates the drift-independent Yang-Zhang volatility proxy.
        
        Args:
            df (pd.DataFrame): Dataframe containing OHLC data.
            window (int): The rolling window period for variance calculation.
            
        Returns:
            pd.Series: Annualized Yang-Zhang volatility series.
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
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        yz_variance = sigma_o_var + (k * sigma_c_var) + ((1 - k) * sigma_rs_var)
        return np.sqrt(yz_variance) * np.sqrt(252)

    @staticmethod
    def get_garch_volatility(returns, model_type='GARCH'):
        """
        Fits a GARCH-family model to returns and extracts conditional volatility.
        
        Args:
            returns (pd.Series): Time series of log returns.
            model_type (str): Type of model to fit ('GARCH', 'EGARCH', or 'TGARCH').
            
        Returns:
            tuple: (Annualized volatility series, dictionary of model parameters).
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
            annualized_vol = daily_vol_decimal * np.sqrt(252)
            
            return annualized_vol, res.params.to_dict()
        except:
            return pd.Series(np.zeros(len(returns)), index=returns.index), {}

class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        """
        Initializes the Soft-Attention layer for dynamic feature weighting.
        
        Args:
            input_dim (int): Number of input features to be weighted.
        """
        super(FeatureAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Computes attention weights based on the temporal mean of the input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).
            
        Returns:
            torch.Tensor: Normalized feature weights.
        """
        context = torch.mean(x, dim=1) 
        weights = self.attn(context)
        return weights

class DualAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initializes the Hybrid LSTM architecture with integrated feature attention.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in the LSTM hidden layer.
        """
        super(DualAttentionLSTM, self).__init__()
        self.feature_attn = FeatureAttention(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.latest_feature_weights = None 
        
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        Forward pass applying feature weighting followed by temporal LSTM processing.
        
        Args:
            x (torch.Tensor): Input sequence tensor.
            
        Returns:
            torch.Tensor: Non-negative volatility prediction.
        """
        f_weights = self.feature_attn(x) 
        self.latest_feature_weights = f_weights.detach()
        
        f_weights_expanded = f_weights.unsqueeze(1)
        weighted_input = x * f_weights_expanded
        
        lstm_out, _ = self.lstm(weighted_input)
        last_hidden = lstm_out[:, -1, :]
        
        raw_out = self.fc(last_hidden)
        # Softplus ensures predicted volatility is always positive
        prediction = F.softplus(raw_out) 
        return prediction

class MGF_LSTM:
    def __init__(self, ticker, df_ticker, df_vix, seq_len=10):
        """
        Orchestrates data preparation, training, and evaluation for a specific asset.
        
        Args:
            ticker (str): Asset identifier.
            df_ticker (pd.DataFrame): Dataframe with OHLC data.
            df_vix (pd.DataFrame): Dataframe with VIX sentiment data.
            seq_len (int): Lookback window for sequence generation.
        """
        self.ticker = ticker
        self.df = df_ticker.copy()
        self.vix = df_vix.copy()
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.model = None
        self.logs = {
            "ticker": ticker,
            "metrics": {},
            "predictions": {},
            "feature_names": [],
            "feature_weights_history": [] 
        }

    def prepare_data(self):
        """
        Performs feature engineering including log returns, YZ volatility, and GARCH fitting.
        
        Returns:
            pd.DataFrame: Fully processed feature set.
        """
        self.df['Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['YZ'] = VolatilityCalculators.calculate_yang_zhang_volatility(self.df)
        self.df.dropna(inplace=True)
        
        returns = self.df['Returns']
        garch, _ = VolatilityCalculators.get_garch_volatility(returns, 'GARCH')
        egarch, _ = VolatilityCalculators.get_garch_volatility(returns, 'EGARCH')
        tgarch, _ = VolatilityCalculators.get_garch_volatility(returns, 'TGARCH')
        
        self.df['GARCH'] = garch
        self.df['EGARCH'] = egarch
        self.df['TGARCH'] = tgarch
        
        self.df = self.df.join(self.vix['Close'].rename("India_VIX"))
        self.df['India_VIX'] = self.df['India_VIX'].ffill()
        
        self.df.dropna(inplace=True)
        return self.df

    def qlike_loss(self, y_pred, y_true):
        """
        Computes the Quasi-Likelihood (QLIKE) loss function.
        
        Args:
            y_pred (torch.Tensor): Predicted volatility.
            y_true (torch.Tensor): Ground truth (YZ) volatility.
            
        Returns:
            torch.Tensor: Calculated loss value.
        """
        y_pred = y_pred + 1e-6 
        loss = 2 * torch.log(y_pred) + (y_true**2 / y_pred**2)
        return torch.mean(loss)

    def train(self, epochs=500):
        """
        Trains the Dual-Attention LSTM using QLIKE loss and evaluates out-of-sample risk.
        
        Args:
            epochs (int): Number of training iterations.
            
        Returns:
            nn.Module: Trained PyTorch model.
        """
        feature_cols = ['YZ', 'GARCH', 'EGARCH', 'TGARCH', 'India_VIX', 'Returns']
        target_col = 'YZ'
        
        data_values = self.df[feature_cols].values
        target_values = self.df[target_col].values

        split = int(0.8 * len(data_values))
        train_data, test_data = data_values[:split], data_values[split:]
        train_target, test_target = target_values[:split], target_values[split:]
        
        self.scaler.fit(train_data)
        scaled_train = self.scaler.transform(train_data)
        scaled_test = self.scaler.transform(test_data)
        
        def create_sequences(data, target):
            X, y = [], []
            for i in range(len(data) - self.seq_len):
                X.append(data[i : i + self.seq_len])
                y.append(target[i + self.seq_len])
            return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(1)

        X_train, y_train = create_sequences(scaled_train, train_target)
        X_test, y_test = create_sequences(scaled_test, test_target)
        
        self.model = DualAttentionLSTM(input_dim=len(feature_cols))
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        print(f"   Training {self.ticker}...")
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
            full_weight_history = self.model.latest_feature_weights.cpu().numpy()
            
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
            
            breach_90 = np.sum(actual_daily_returns < var_90)
            breach_95 = np.sum(actual_daily_returns < var_95)
            breach_99 = np.sum(actual_daily_returns < var_99)
            total_days = len(actual_daily_returns)
            
            self.logs['feature_names'] = feature_cols
            self.logs['feature_weights_history'] = full_weight_history.tolist()
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
                "Breach_Ratio_90": float(breach_90 / total_days),
                "Breach_Ratio_95": float(breach_95 / total_days),
                "Breach_Ratio_99": float(breach_99 / total_days)
            }
            
            print(f"   [RESULT] RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | Breach 95%: {(breach_95/total_days)*100:.2f}%")

        return self.model