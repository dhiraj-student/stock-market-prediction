# 📈 Stock Market Prediction Using LSTM

This project focuses on predicting the **Nifty 50** stock index using a deep learning model—**LSTM (Long Short-Term Memory)**. The model is trained on one year of historical stock price data to predict future trends. This work demonstrates how machine learning and deep learning can be applied to time-series forecasting in the finance domain.

##  Objective
- To predict stock prices based on historical data
- To apply LSTM, a deep learning model effective in sequence-based data
- To evaluate model performance using prediction vs actual price plots

## Machine Learning Approach

## Model: LSTM (Long Short-Term Memory)

- **Why LSTM?** LSTM is a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies, making it ideal for stock price time-series forecasting.
- **Architecture:**
  - Memory cells to retain information
  - Input, Output, and Forget gates for state control
  - Fully connected Dense layer after LSTM layers for final output

## Dataset
- **Source:** Historical stock prices of **Nifty 50** index
- **Time Period:** July 12, 2023 – July 13, 2024
- **Features Used:** Only `Close` prices were used for training and prediction.

# Data Preprocessing
- Missing values handled using `dropna()`
- Normalization performed using `MinMaxScaler` for scaling data to [0, 1]
- Used a sliding window of 60 days to predict the 61st day's price
- Dataset split into training and testing portions

##  Model Building
```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predicted price
