#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[3]:


import yfinance as yf
import streamlit as st

# Streamlit title and instructions
st.title("Indian Stock Data Retriever")
st.write("Enter the stock symbol of an Indian company (e.g., INFY.NS) to retrieve historical data.")

# Stock symbol input
stock_symbol = st.text_input("Stock Symbol", value="INFY.NS")  # Default to "INFY.NS" for user convenience

# Download data on button click
if st.button("Get Data"):
    try:
        # Download historical data from Yahoo Finance
        data = yf.download(stock_symbol, start="2015-01-01")
        
        # Display the data
        st.write(f"Historical data for {stock_symbol}:")
        st.dataframe(data.head())
        
        # Save the data to a CSV file
        csv_filename = f"{stock_symbol}_historical_data.csv"
        data.to_csv(csv_filename)
        
        # Provide a download link for the CSV file
        st.success(f"Data saved to {csv_filename}.")
        with open(csv_filename, 'rb') as file:
            st.download_button(
                label="Download CSV",
                data=file,
                file_name=csv_filename,
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error retrieving data: {e}")


# In[4]:


df= pd.read_csv(fr"C:\Users\Manan\Downloads\{stock_symbol}_historical_data.csv")


# In[5]:


df = df.drop([0,1]).reset_index(drop=True)# Drop the first row (misaligned row)


# In[6]:


df.columns = ["Date", "Price", "Close", "High", "Low", "Open", "Volume"]  
df.reset_index(drop=True, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
df[['Price', 'Close', 'High', 'Low', 'Open', 'Volume']] = df[['Price', 'Close', 'High', 'Low', 'Open', 'Volume']].astype(float)
print(df.head())


# In[7]:


df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
X = df[['Open-Close', 'High-Low']]
X.head()


# In[8]:


y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
y


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=42)


# In[10]:


cls = SVC().fit(X_train, y_train)


# In[19]:


df['Predicted_Signal'] = cls.predict(X)


# In[20]:


df['Return'] = df.Close.pct_change()


# In[21]:


df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)


# In[25]:


df['Cum_Ret'] = df['Return'].cumsum()
df


# In[27]:


df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
df


# In[29]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df['Cum_Ret'], color='red', label='Cumulative Return')
ax.plot(df['Cum_Strategy'], color='blue', label='Cumulative Strategy')
ax.set_title("Cumulative Return vs Strategy")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)


# In[30]:


A = ((df['Cum_Strategy'] - df['Cum_Ret']) / df['Cum_Ret']) * 100
mean = np.mean(A)
st.write("Mean Percentage Difference:")
st.metric(label="Mean Difference (%)", value=f"{mean:.2f}")


# In[33]:


y_pred = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
latest_data = X.iloc[-1:].values.reshape(1, -1)  # Reshape if needed for single prediction
prediction = cls.predict(latest_data)
st.write(f"Prediction for tomorrow: {'Up' if prediction[0] == 1 else 'Down'}")


# In[ ]:




