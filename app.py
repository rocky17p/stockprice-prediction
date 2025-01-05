import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('stock_model.keras')

# Streamlit app title
st.title('📈 Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Fetch stock data
data = yf.download(stock)

# Show stock data
st.subheader('Stock Data')
st.write(data.tail())

# Moving average
ma_100 = data['Close'].rolling(100).mean()

# Plot MA and closing price
st.subheader('Closing Price vs 100-Day MA')
fig1 = plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Closing Price')
plt.plot(ma_100, label='100-Day MA', color='red')
plt.legend()
st.pyplot(fig1)

# Prepare test data
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)

# Use pd.concat instead of append
final_data = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale the data
input_data = scaler.fit_transform(final_data)

# Prepare input data for prediction
x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot predictions
st.subheader('Original Price vs Predicted Price')
fig2 = plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Original Price')
plt.plot(y_predicted, label='Predicted Price', color='red')
plt.legend()
st.pyplot(fig2)

# Function to predict the future stock price for 5 days
# Function to predict the future stock price for 5 days
# Function to predict the future stock price for 5 days
def predict_future_price(data, model, scaler, days_to_predict):
    # Get the last 100 days of stock data
    last_100_days = data.tail(100)
    
    # Prepare the last 100 days for scaling
    scaled_last_100 = scaler.transform(last_100_days.values.reshape(-1, 1))
    
    # Prepare the input for prediction
    x_input = scaled_last_100.reshape(1, 100, 1)  # shape (1, 100, 1)
    
    # Predict future prices
    future_predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(x_input)
        future_predictions.append(pred[0][0])
        # Reshape pred to make sure it's 3D before appending
        pred_reshaped = pred.reshape(1, 1, 1)
        # Update the input for the next prediction (roll forward)
        x_input = np.append(x_input[:, 1:, :], pred_reshaped, axis=1)
    
    # Reverse the scaling for the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions



# Predict future price for 5 days
if st.button('Predict Price for Next 5 Days'):
    future_price = predict_future_price(data['Close'], model, scaler, 5)
    
    # Display the predicted future prices
    st.subheader('Predicted Price for the Next 5 Days')
    st.write(future_price)

    # Plot predicted future prices
    st.subheader('Predicted Prices for the Next 5 Days')
    future_days = np.array([f'Day {i+1}' for i in range(5)])
    
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(future_days, future_price, label='Predicted Price', marker='o', color='orange')
    plt.title(f'Predicted Stock Prices for {stock} - Next 5 Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)
