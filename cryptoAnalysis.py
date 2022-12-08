# Filename: cryptoAnalysis.py
# Description: This python program is used to analyse and compare the crypto currency data

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.colors as col
from scipy.stats import pearsonr


# This class is used to perform crypto analysis on data
class CryptoAnalysis:

    def buildPearsonCoefficient(self,data):
        pearson_corr = data.corr(method='pearson')
        pearson_corr.round(2).style.background_gradient(cmap='coolwarm')        
                
        plt.figure(figsize = (10,10))
        ax = plt.axes()
        ax.set_title('Correlation for the Top Cryptocurrencies',fontweight="bold",size=16,color='#884EA0')
        sns.heatmap(data.corr(method='pearson'),annot=True, cmap='Blues',ax = ax)        
        plt.show()
        # pearson correlation
    
    
    def compareCryptoPrices(self,crypto_df):
        #print(crypto_df.head(5))
        title_color = '#884EA0'
        #Compare bit coin price to other crypto currencies        
        fig = px.line(crypto_df, y=['btc','eth','usdt','usdc','bnb'],title="Comparison of Cryptocurrencies between 2021-2022",
            labels={"value": "Closing Price (CAD)", "variable": "Cryptocurrency"} )
        fig.show()

        # BTC PLOT 
        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.subplot(3, 2, 1)
        plt.plot(crypto_df['btc'])
        # Set title and labels for the plot
        plt.title('Bitcoin', fontsize=10, color='#CD5C5C')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Closing Price (CAD)', fontsize=10)
        plt.tick_params(axis='both', labelsize=10)
        #ETH
        plt.subplot(3, 2, 2)
        plt.plot(crypto_df['eth'])
        # Set title and labels for the plot
        plt.title('Ethereum', fontsize=10, color='#CD5C5C')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Closing Price (CAD)', fontsize=10)
        plt.tick_params(axis='both', labelsize=10)
        #USDT
        plt.subplot(3, 2, 3)
        plt.plot(crypto_df['usdt'])
        # Set title and labels for the plot
        plt.title('Tether', fontsize=10, color='#CD5C5C')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Closing Price (CAD)', fontsize=10)
        plt.tick_params(axis='both', labelsize=10)
        #USDC
        plt.subplot(3, 2, 4)
        plt.plot(crypto_df['usdc'])        
        # Set title and labels for the plot
        plt.title('US Dollar Coin', fontsize=10, color='#CD5C5C')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Closing Price (CAD)', fontsize=10)
        plt.tick_params(axis='both', labelsize=10)
        #BNB
        plt.subplot(3, 2, 5)
        plt.plot(crypto_df['bnb'])
        # Set title and labels for the plot
        plt.title('Binance Coin', fontsize=10, color='#CD5C5C')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Closing Price (CAD)', fontsize=10)
        plt.tick_params(axis='both', labelsize=10)

        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=None)
        # TO CHECK
        plt.suptitle('Comparison of Bitcoin with other Cryptocurrencies', fontweight="bold",size=18,color=title_color)
        plt.show()

        # compare the price with RSI indicator
        # Create a Price Chart on BTC and ETH
        x = crypto_df.index
        fig, ax0 = plt.subplots(figsize=(16, 8), sharex=False)

        # Price Chart for BTC-CAD Close
        color_code = 'tab:blue'
        y = crypto_df['btc']
        ax0.set_xlabel('Date')
        ax0.set_ylabel('BTC-Close (CAD)', color=color_code, fontsize=14)
        ax0.plot(x, y, color=color_code)
        ax0.tick_params(axis='y', labelcolor=color_code)
        ax0.text(0.02, 0.95, 'BTC-CAD',  transform=ax0.transAxes, color=color_code, fontsize=14)

        # Price Chart for ETH-CAD Close
        color_code = 'tab:red'
        y = crypto_df['eth']
        # enable twin axis
        ax1 = ax0.twinx()
        # x label handled with ax1
        ax1.set_ylabel('ETH-Close (CAD)', color=color_code, fontsize=14)  
        ax1.plot(x, y, color=color_code)
        ax1.tick_params(axis='y', labelcolor=color_code)
        ax1.text(0.02, 0.9, 'ETH-CAD',  transform=ax1.transAxes, color=color_code, fontsize=14)
        plt.suptitle('Closing Prices of Bitcoin and Ethereum for the last 5 years', fontweight="bold",size=16,color=title_color)
        plt.show()

    # add rsi indicators
    def addRsiIndicators(self,df):
        # Calculate the RSI
        # Moving Averages on high, lows, and std - different periods
        df['MA200_low'] = df['low-btc'].rolling(window=200).min()
        df['MA14_low'] = df['low-btc'].rolling(window=14).min()
        df['MA200_high'] = df['high-btc'].rolling(window=200).max()
        df['MA14_high'] = df['high-btc'].rolling(window=14).max()

        # Relative Strength Index (RSI)
        # k ratio formula = [100 * (close-low)] / (high-low)
        df['K-ratio'] = 100*((df['close-btc'] - df['MA14_low']) / (df['MA14_high'] - df['MA14_low']) )
        df['RSI'] = df['K-ratio'].rolling(window=3).mean() 

        # Replace nas 
        df.fillna(0, inplace=True)    

        # function that converts a given set of indicator values to colors
        def get_colors_indicators(ind, colormap):
            colorlist = []
            norm = col.Normalize(vmin=ind.min(), vmax=ind.max())
            for i in ind:
                colorlist.append(list(colormap(norm(i))))
            return colorlist

        # convert the RSI values                         
        y = np.array(df['RSI'])
        colormap = plt.get_cmap('plasma')
        df['rsi_colors'] = get_colors_indicators(y, colormap)
        
        # Create a coloured price chart for BTC-CAD
        pd.plotting.register_matplotlib_converters()
        fig, ax1 = plt.subplots(figsize=(14, 10), sharex=False)
        x = df.index
        y = df['close-btc']
        z = df['rsi_colors']
        #color_code = 'tab:blue'
        ax2 = ax1.twinx()

        # draw points
        for i in range(len(df)):
            ax1.plot(x[i], np.array(y[i]), 'o', color=z[i], alpha = 0.5, markersize=5)
        ax1.set_ylabel('BTC-Close(CAD)')
        ax1.set_xlabel('Date')
        ax1.tick_params(axis='y', labelcolor='black')        
        ax1.text(0.02, 0.95, 'BTC-CAD - RSI', transform=ax1.transAxes, fontsize=14)

        # plot the color bar
        pos_clip = ax2.imshow(list(z), cmap='plasma',  interpolation='none', vmin=0, vmax=100)
        cb = plt.colorbar(pos_clip)
        plt.show()

    # This method will add technical indicators and print the chart for the crypto symbol  
    def technicalIndicators(self,df):
        def add_indicators(df):
            # Calculate the 30 day Pearson Correlation 
            cor_period = 30 #this corresponds to a monthly correlation period
            columntobeadded = [0] * cor_period
            df = df.fillna(0) 
            for i in range(len(df)-cor_period):
                btc = df['close-BTC'][i:i+cor_period]
                eth = df['close-ETH'][i:i+cor_period]
                corr, _ = pearsonr(btc, eth)
                columntobeadded.append(corr)    
            # insert the colours into our original dataframe    
            df.insert(2, "P_Correlation", columntobeadded, True)

            # Calculate the RSI
            # Moving Averages on high, lows, and std - different periods
            df['MA200_low'] = df['low-BTC'].rolling(window=200).min()
            df['MA14_low'] = df['low-BTC'].rolling(window=14).min()
            df['MA200_high'] = df['high-BTC'].rolling(window=200).max()
            df['MA14_high'] = df['high-BTC'].rolling(window=14).max()

            # Relative Strength Index (RSI)
            df['K-ratio'] = 100*((df['close-BTC'] - df['MA14_low']) / (df['MA14_high'] - df['MA14_low']) )
            df['RSI'] = df['K-ratio'].rolling(window=3).mean() 

            # Replace nas             
            df.fillna(0, inplace=True)    
            return df
        
        dfcr = add_indicators(df)
        color = 'tab:red'
        # Visualize measures
        fig, ax1 = plt.subplots(figsize=(22, 4), sharex=False)
        plt.ylabel('ETH-BTC Price Correlation', color=color)  # we already handled the x-label with ax1
        x = y = dfcr.index
        ax1.plot(x, dfcr['P_Correlation'], color='black')
        ax2 = ax1.twinx()
        ax2.plot(x, dfcr['RSI'], color='blue')
        plt.tick_params(axis='y', labelcolor=color)

        plt.show()

        # function that converts a given set of indicator values to colors
        def get_colors(ind, colormap):
            colorlist = []
            norm = col.Normalize(vmin=ind.min(), vmax=ind.max())
            for i in ind:
                colorlist.append(list(colormap(norm(i))))
            return colorlist

        # convert the RSI                         
        y = np.array(dfcr['RSI'])
        colormap = plt.get_cmap('plasma')
        dfcr['rsi_colors'] = get_colors(y, colormap)

        # convert the Pearson Correlation
        y = np.array(dfcr['P_Correlation'])
        colormap = plt.get_cmap('plasma')
        dfcr['cor_colors'] = get_colors(y, colormap)

        # Creating a Price Chart
        pd.plotting.register_matplotlib_converters()
        fig, ax1 = plt.subplots(figsize=(18, 10), sharex=False)
        x = dfcr.index
        y = dfcr['close-BTC']
        z = dfcr['rsi_colors']

        # drawing the points
        for i in range(len(dfcr)):
           ax1.plot(x[i], np.array(y[i]), 'o',  color=z[i], alpha = 0.5, markersize=5)
        ax1.set_ylabel('BTC-Close (CAD)')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlabel('Date')
        ax1.text(0.02, 0.95, 'BTC-CAD - By RSI',  transform=ax1.transAxes, fontsize=16)

        # plotting the color bar
        pos_neg_clipped = ax2.imshow(list(z), cmap='plasma', vmin=0, vmax=100, interpolation='none')
        cb = plt.colorbar(pos_neg_clipped)
        plt.show()
        
        
