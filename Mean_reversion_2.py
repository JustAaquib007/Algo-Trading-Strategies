#!/usr/bin/env python
# coding: utf-8

# In[86]:


import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

df = yf.download("TSLA",'2019-01-01')

ma_days= 20
ema_days = 5

df['ma'] = df.Close.rolling(ma_days).mean() # days MA
df['vol'] = df.Close.rolling(ma_days).std()  # MA standard deviation

df['ema']  = df.Close.ewm(span=ema_days, adjust=False).mean() # days EMA
df['vol'] = df.Close.rolling(ema_days).std()  # EMA standard deviation

df['ma_upper_b'] = df.ma + (2*df.vol)  # upper ma bolinger
df['ma_lower_b'] = df.ma - (2*df.vol)  # lower ma bolinger

df['ema_upper_b'] = df.ma + (2*df.vol)  # upper ema bolinger
df['ema_lower_b'] = df.ma - (2*df.vol)  # lower ema bolinger


# In[87]:


# Calculating RSI indicator
# RSI = average gain / average loss "over a period of time"
df['rsi'] = ta.momentum.rsi(df.Close, window = 6)


# In[88]:


# crafting conditions
ma_conditions = [
                (df.rsi < 30) & (df.Close < df.ma_lower_b), #oversold
                (df.rsi > 70) & (df.Close > df.ma_lower_b)  #overbought
]

ema_conditions = [
                (df.rsi < 30) & (df.Close < df.ema_lower_b), #oversold
                (df.rsi > 70) & (df.Close > df.ema_lower_b)  #overbought
]

Choices = ["buy","sell"]

df['ma_trade_signal'] = np.select(ma_conditions, Choices)
df['ema_trade_signal'] = np.select(ema_conditions, Choices)

df.dropna(inplace = True) # drop rows with missing values

df.ma_trade_signal = df.ma_trade_signal.shift()
df.ema_trade_signal = df.ema_trade_signal.shift()


# In[89]:


#placing trades

Long = False # no trades yet

buydates, selldates = [], []
buyprices, sellprices = [],[]

for index, row in df.iterrows():
    if not Long and row['ma_trade_signal'] == "buy":
        buydates.append(index)
        buyprices.append(row.Open)
        Long = True 
    if Long:
        if row['ma_trade_signal'] == 'sell' or row.Close < (0.95*buyprices[-1]):
            selldates.append(index)
            sellprices.append(row.Open)
            Long = False
        


# In[90]:


plt.plot(df.Close)
plt.scatter(df.loc[buydates].index, df.loc[buydates].Close, marker = '^', c='g')
plt.scatter(df.loc[selldates].index, df.loc[selldates].Close, marker = 'v', c='r')


# In[91]:


# Calculating Profit and Losses
statements = [(sell - buy) / buy for sell,buy in zip(sellprices, buyprices)] 

print("Mean Reversion Performance")
print("{} number of trades".format(len(statements)))

Capital = 1000000
res = ( pd.Series( [(sell-buy)/buy for sell,buy in zip(sellprices,buyprices)] ) + 1).prod() -1
print(round(res*100,2),"%")

for i in statements:
    Capital = Capital*(i+1)

print("$",round(Capital,2),"\n")    

#------------------------------------ASSET PERFORMANCE----------------------------------------
eq_return = ((df.Close[-1]-df.Close[0])/df.Close[0])*100
print("Equity Performance")
print(round(eq_return,2),"% \n")

#--------------------------------------BENCHMARK-----------------------------------------------
SP = yf.download("^GSPC",'2019-01-01')

SP_return = ((SP.Close[-1]-SP.Close[0])/SP.Close[0])*100
print("S&P 500 Performance")
print("S&P 500 returned",round(SP_return,2),"%",)


# In[ ]:





# In[ ]:





# In[ ]:




