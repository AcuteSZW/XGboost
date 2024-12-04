# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import logging
import os
import tempfile
from skopt import BayesSearchCV
from scipy.stats import uniform, loguniform

# 设置临时文件夹路径为纯 ASCII 路径
tempfile.tempdir = 'C:\\Temp'
os.environ['TMPDIR'] = tempfile.tempdir
os.environ['TEMP'] = tempfile.tempdir
os.environ['TMP'] = tempfile.tempdir

# 确保临时文件夹存在
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir)

os.environ['PYTHONIOENCODING'] = 'utf-8'  # 设置环境变量以确保默认编码为 UTF-8
iter = 24  # 迭代次数
jobs = 6  # 启动核心数

# 配置日志记录，确保日志文件也是 UTF-8 编码
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bollinger_width = upper_band - lower_band
    return upper_band, rolling_mean, lower_band, bollinger_width

def read_and_prepare_data(file_path):
    logging.info("1. 读取并准备数据")
    if not os.path.exists(file_path):
        logging.error(f"文件 {file_path} 不存在，请检查路径。")
        exit(1)
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        logging.error(f"文件 {file_path} 编码错误，请确保文件是 UTF-8 编码。")
        exit(1)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    return data

def compute_technical_indicators(data):
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['RSI'] = compute_rsi(data)
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'], data['Bollinger_Width'] = compute_bollinger_bands(data)
    data['Close_diff'] = data['Close'].diff().fillna(0)  # 填充第一个 Close_diff 的 NaN 为 0

    # 添加更多技术指标
    data['MACD_Line'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    data['Momentum'] = data['Close'].pct_change(periods=5) * 100  # 5天动量

    # 添加滞后特征
    data['Close_diff_2'] = data['Close_diff'].shift(1).fillna(0)
    data['Close_diff_3'] = data['Close_diff'].shift(2).fillna(0)

    # 添加季节性特征
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter

    # 删除 NaN 值
    data.dropna(inplace=True)
    return data

def split_train_test(data, features, target):
    logging.info("3. 划分训练集和测试集")
    train_size = int(len(data) * 0.80)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]
    return X_train, y_train, X_test, y_test

def feature_scaling(X_train, X_test):
    logging.info("4. 特征缩放")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def hyperparameter_optimization(X_train_scaled, y_train, tscv, jobs, iter):
    param_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    param_gb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'subsample': [0.8, 0.9, 1.0]
    }

    # param_dist_xgb = {
    #     'n_estimators': (100, 300),
    #     'learning_rate': (0.01, 0.1, 'log-uniform'),
    #     'max_depth': (3, 10),
    #     'min_child_weight': (1, 10),
    #     'gamma': (0, 0.3, 'uniform'),
    #     'subsample': (0.8, 1.0, 'uniform'),
    #     'colsample_bytree': (0.8, 1.0, 'uniform')
    # }

    param_dist_xgb = {
    'n_estimators': range(100, 301),  # 注意这里使用range以确保整数值
    'learning_rate': loguniform(0.01, 0.1),
    'max_depth': range(3, 11),  # 注意这里使用range以确保整数值
    'min_child_weight': range(1, 11),  # 注意这里使用range以确保整数值
    'gamma': uniform(0, 0.3),
    'subsample': uniform(0.8, 0.2),
    'colsample_bytree': uniform(0.8, 0.2)
    }

    rf_model = RandomForestRegressor(random_state=42)
    gb_model = GradientBoostingRegressor(random_state=42)
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

    models = [
        ('RandomForest', rf_model, param_rf),
        ('GradientBoosting', gb_model, param_gb),
        ('XGBoost', xgb_model, param_dist_xgb)
    ]

    best_models = {}
    for name, model, params in models:
        logging.info(f"5. 超参数优化 - {name}")
        if isinstance(params, dict):
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=iter,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=jobs,
                random_state=42,
                error_score=np.nan
            )
        else:
            random_search = BayesSearchCV(
                estimator=model,
                search_spaces=params,
                n_iter=iter,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=jobs,
                random_state=42,
                verbose=1
            )
        
        try:
            random_search.fit(X_train_scaled, y_train)
            best_params = random_search.best_params_
            logging.info(f"{name}最佳参数: {best_params}")
            best_models[name] = random_search.best_estimator_
        except Exception as e:
            logging.error(f"在拟合{name}模型时发生错误: {e}")
            exit(1)
    return best_models

def ensemble_predict(models, X):
    predictions = np.array([model.predict(X) for model in models.values()])
    return np.mean(predictions, axis=0)

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logging.info(f"新的 RMSE: {rmse}")
    logging.info(f"新的 MAE: {mae}")
    logging.info(f"R²: {r2}")
    return rmse, mae, r2

def plot_predictions(y_true, y_pred, title):
    logging.info(title)
    plt.figure(figsize=(14, 7))
    
    if not y_true.empty:
        plt.plot(y_true.index, y_true, label='Actual', color='blue')
    
    plt.plot(y_pred.index, y_pred, label='Predicted', color='orange', linestyle='--')
    
    if not y_true.empty:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        y_pred_lower = y_pred - rmse
        y_pred_upper = y_pred + rmse
        plt.fill_between(y_true.index, y_pred_lower, y_pred_upper, color='gray', alpha=0.3, label='±1 RMSE')
        
        y_pred_lower_2 = y_pred - 2 * rmse
        y_pred_upper_2 = y_pred + 2 * rmse
        plt.fill_between(y_true.index, y_pred_lower_2, y_pred_upper_2, color='lightgray', alpha=0.3, label='±2 RMSE')
        
        plt.title(f"{title}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    else:
        plt.title(title)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def predict_future(best_models, scaler, future_dates, last_features, features, data):
    logging.info("创建未来日期的数据框（12月1日至明年1月1日）")
    future_data = pd.DataFrame(index=future_dates)
    for col in last_features.index:
        future_data[col] = last_features[col]

    logging.info("10. 逐步更新未来的特征并进行预测")
    for i in range(len(future_data)):
        if i == 0:
            prev_features = last_features.values.reshape(1, -1)
        else:
            prev_features = future_data.iloc[i-1][features].values.reshape(1, -1)
        
        predicted_close = ensemble_predict(best_models, scaler.transform(pd.DataFrame(prev_features, columns=features)))[0]
        future_data.loc[future_data.index[i], 'Close'] = predicted_close
        
        combined_data = pd.concat([data.tail(20), future_data.head(i+1)])
        future_data.loc[future_data.index[i], 'SMA_5'] = combined_data['Close'].rolling(window=5, min_periods=1).mean().iloc[-1]
        future_data.loc[future_data.index[i], 'SMA_20'] = combined_data['Close'].rolling(window=20, min_periods=1).mean().iloc[-1]
        future_data.loc[future_data.index[i], 'EMA_5'] = combined_data['Close'].ewm(span=5, adjust=False).mean().iloc[-1]
        future_data.loc[future_data.index[i], 'RSI'] = compute_rsi(combined_data).iloc[-1]
        upper_band, _, lower_band, bollinger_width = compute_bollinger_bands(combined_data)
        future_data.loc[future_data.index[i], 'Bollinger_Upper'] = upper_band.iloc[-1]
        future_data.loc[future_data.index[i], 'Bollinger_Middle'] = lower_band.iloc[-1]
        future_data.loc[future_data.index[i], 'Bollinger_Lower'] = lower_band.iloc[-1]
        future_data.loc[future_data.index[i], 'Bollinger_Width'] = bollinger_width.iloc[-1]
        future_data.loc[future_data.index[i], 'Close_diff'] = future_data['Close'].diff().fillna(0).iloc[i]
        future_data.loc[future_data.index[i], 'MACD_Line'] = combined_data['Close'].ewm(span=12, adjust=False).mean().iloc[-1] - combined_data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
        future_data.loc[future_data.index[i], 'MACD_Signal'] = future_data['MACD_Line'].ewm(span=9, adjust=False).mean().iloc[-1]
        future_data.loc[future_data.index[i], 'MACD_Histogram'] = future_data['MACD_Line'].iloc[-1] - future_data['MACD_Signal'].iloc[-1]
        future_data.loc[future_data.index[i], 'Momentum'] = combined_data['Close'].pct_change(periods=5).fillna(0).iloc[-1] * 100
        future_data.loc[future_data.index[i], 'Close_diff_2'] = future_data['Close_diff'].shift(1).fillna(0).iloc[i]
        future_data.loc[future_data.index[i], 'Close_diff_3'] = future_data['Close_diff'].shift(2).fillna(0).iloc[i]
        future_data.loc[future_data.index[i], 'Day_of_Week'] = future_data.index[i].dayofweek
        future_data.loc[future_data.index[i], 'Month'] = future_data.index[i].month
        future_data.loc[future_data.index[i], 'Quarter'] = future_data.index[i].quarter

    logging.info("11. 使用模型预测未来的价格")
    future_predictions = ensemble_predict(best_models, scaler.transform(future_data[features]))
    future_data['Predicted_Close'] = future_predictions
    return future_data

def main():
    logging.info(f"=======================以下迭代 {iter} 次，启动 {jobs} 核心。 =============================================")
    file_path = 'shanghai_composite_index_2000_01_01_to_2024-12-30.csv'
    data = read_and_prepare_data(file_path)
    data = compute_technical_indicators(data)
    
    features = ['SMA_5', 'SMA_20', 'EMA_5', 'RSI', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower', 
                'Bollinger_Width', 'Close_diff', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram', 'Momentum', 
                'Close_diff_2', 'Close_diff_3', 'Day_of_Week', 'Month', 'Quarter']
    target = 'Close'

    X_train, y_train, X_test, y_test = split_train_test(data, features, target)
    X_train_scaled, X_test_scaled, scaler = feature_scaling(X_train, X_test)
    tscv = TimeSeriesSplit(n_splits=5)
    best_models = hyperparameter_optimization(X_train_scaled, y_train, tscv, jobs, iter)

    y_pred = ensemble_predict(best_models, X_test_scaled)
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    rmse, mae, r2 = evaluate_model(y_test, y_pred_series)
    plot_predictions(y_test, y_pred_series, "Shanghai Composite Index Prediction (Optimized Model)")

    future_dates = pd.date_range(start='2024-12-05', end='2025-01-01', freq='D')
    last_features = data.iloc[-1][features].ffill()
    future_data = predict_future(best_models, scaler, future_dates, last_features, features, data)
    
    # 获取实际的价格值（如果有）
    actual_prices = data.loc[future_dates.intersection(data.index), 'Close'].dropna()
    predicted_prices = future_data.loc[future_dates.intersection(future_data.index), 'Predicted_Close']
    
    if not actual_prices.empty and not predicted_prices.empty:
        plot_predictions(actual_prices, predicted_prices, "Shanghai Composite Index Prediction (Dec 1, 2024 to Jan 1,2025)")
    elif not predicted_prices.empty:
        logging.warning("没有足够的实际价格数据来绘制未来预测图表，仅显示预测数据。")
        plot_predictions(pd.Series(), predicted_prices, "Shanghai Composite Index Prediction (Dec 1, 2024 to Jan 1,2025)")
    else:
        logging.warning("没有足够的数据来绘制未来预测图表。")
    
    logging.info("结束")

if __name__ == "__main__":
    main()


