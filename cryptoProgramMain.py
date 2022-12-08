# File Name: cryptoProgramMain.py
# Description: This is the main program to perform the cryptocurrency analysis and prediction

# import the modules - python program
from commonUtils import CommonUtils
from cryptoAnalysis import CryptoAnalysis
from models import Models
from backTesting import BackTesting
# import the required python libraries
import pandas as pd
import numpy as np
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

class CryptoProgramMain:
   

    def performCryptoAnalysisTasks():
        option1 = 1
        option2 = 2
        option3 = 3
        option4 = 4
        option5 = 5
        # Get the DB Connection
        con = CommonUtils.getDBConnection()
        try:
            with con.cursor() as cursor:
                con.select_db("CRYPTO_CURRENCY")
                SQL_select_data = "select * from CRYPTO_CURRENCY_DATA"
                cursor.execute(SQL_select_data)
                crypto_data = cursor.fetchall()
                crypto_data_df = pd.DataFrame(crypto_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits', 'Crypto_Symbol', 'Creation_Time'])
                
                
            print("****************************************************************")
            print("****************************************************************")
            print("********Cryptocurrency Price Prediction using ML and DL*********")
            print("****************************************************************")
            print("****************************************************************")
            print("Choose from below Options:")
            print("Option 1: Pearson correlation coefficient for Cryptocurrency data")
            print("Option 2: Comparison of Bitcoin to other cryptocurrencies")
            print("Option 3: Backtesting of cryptocurrency data")
            print("Option 4: Build FBProphet Model")
            print("Option 5: Build LSTM Model")

            option = int(input("Please enter an option to perform the research: "))
            if (option == option1):
                # call pearson correlation coefficient
                corr_df = pd.DataFrame(columns = ['btc','eth','usdt','usdc','bnb','xrp','ada','sol','doge'])
                crypto_data_df['Date'] = pd.to_datetime(crypto_data_df['Date'], format='%Y-%m-%d')

                crypto_data_df = crypto_data_df[(crypto_data_df['Date'] >= "2021-01-01") & (crypto_data_df['Date'] <= "2022-10-30")]
                #print(crypto_data_df.shape)
                crypto_data_df = crypto_data_df.dropna()
                #print("After dropping na")
                #print(crypto_data_df.shape)
                crypto_data_df = crypto_data_df.set_index('Date')
                btc_trend_data = crypto_data_df.query("Crypto_Symbol == 'BTC'")
                eth_trend_data = crypto_data_df.query("Crypto_Symbol == 'ETH'")
                usdt_trend_data = crypto_data_df.query("Crypto_Symbol == 'USDT'")
                usdc_trend_data = crypto_data_df.query("Crypto_Symbol == 'USDC'")
                bnb_trend_data = crypto_data_df.query("Crypto_Symbol == 'BNB'")
                xrp_trend_data = crypto_data_df.query("Crypto_Symbol == 'XRP'")
                ada_trend_data = crypto_data_df.query("Crypto_Symbol == 'ADA'")
                sol_trend_data = crypto_data_df.query("Crypto_Symbol == 'SOL'")
                doge_trend_data = crypto_data_df.query("Crypto_Symbol == 'DOGE'")
                trx_trend_data = crypto_data_df.query("Crypto_Symbol == 'TRX'")
                
                                
                corr_df['btc'] = btc_trend_data['Close']
                corr_df['eth'] = eth_trend_data['Close']
                corr_df['usdt'] = usdt_trend_data['Close']
                corr_df['usdc'] = usdc_trend_data['Close']
                corr_df['bnb'] = bnb_trend_data['Close']
                corr_df['xrp'] = xrp_trend_data['Close']
               
                corr_df['ada'] = ada_trend_data['Close']
                corr_df['sol'] = sol_trend_data['Close']
                corr_df['doge'] = doge_trend_data['Close']
                corr_df['trx'] = trx_trend_data['Close']
                              
                # scaling
                logret_df = np.log(corr_df.pct_change()+1)                                   
                
                cryptoObj = CryptoAnalysis()
                pd.set_option('display.max_columns', None)
                
                cryptoObj.buildPearsonCoefficient(logret_df)
                
                
            elif (option == option2):
                # prepare the data for comparing the bitcoin to other crypto currencies
                crypto_data_df['Date'] = pd.to_datetime(crypto_data_df['Date'], format='%Y-%m-%d')

                crypto_data_df = crypto_data_df[(crypto_data_df['Date'] >= "2017-01-01") & (crypto_data_df['Date'] <= "2022-10-30")]
                #print(crypto_data_df.shape)
                crypto_data_df = crypto_data_df.dropna()
                crypto_data_df = crypto_data_df.set_index('Date')
                btc_trend_data = crypto_data_df.query("Crypto_Symbol == 'BTC'")
                eth_trend_data = crypto_data_df.query("Crypto_Symbol == 'ETH'")
                
                usdt_trend_data = crypto_data_df.query("Crypto_Symbol == 'USDT'")
                usdc_trend_data = crypto_data_df.query("Crypto_Symbol == 'USDC'")
                bnb_trend_data = crypto_data_df.query("Crypto_Symbol == 'BNB'")
                
                crypto_df = pd.DataFrame(columns = ['btc','eth','usdt','usdc','bnb'])
                crypto_df['btc'] = btc_trend_data['Close']
                crypto_df['eth'] = eth_trend_data['Close']
                crypto_df['usdt'] = usdt_trend_data['Close']
                crypto_df['usdc'] = usdc_trend_data['Close']
                crypto_df['bnb'] = bnb_trend_data['Close']

                # compare bitcoin to other crypto currencies                
                cryptoObj = CryptoAnalysis()
                cryptoObj.compareCryptoPrices(crypto_df)
                print("2")
                # calculate the rsi indicators and compare the prices
                crypto_df = pd.DataFrame(columns = ['btc-high','btc-low','btc-close'])
                btc_data = crypto_data_df.query("Crypto_Symbol == 'BTC'")
                crypto_df['high-BTC'] = btc_data['High']
                crypto_df['low-BTC'] = btc_data['Low']
                crypto_df['close-BTC'] = btc_data['Close']
                eth_data = crypto_data_df.query("Crypto_Symbol == 'ETH'")
                crypto_df['high-ETH'] = eth_data['High']
                crypto_df['low-ETH'] = eth_data['Low']
                crypto_df['close-ETH'] = eth_data['Close']
                cryptoObj.technicalIndicators(crypto_df)
                #cryptoObj.addRsiIndicators(crypto_df)
                                
            elif (option == option3):
                # Backtesting strategy
                print("Backing Testing Cryptocurrency data....")
                crypto_data_df['Date'] = pd.to_datetime(crypto_data_df['Date'], format='%Y-%m-%d')                
                backTesting = BackTesting()
                backTesting.performBackTesting(crypto_data_df)
            elif (option == option4 or option == option5):
                # call FBProphet Model
                model = Models()
                symbol = input("Enter the Cryptocurrency symbol (BTC/ETH): ")
                symbol_desc = ''
                with con.cursor() as cursor:
                    con.select_db("CRYPTO_CURRENCY")
                    SQL_select_data = "select CONCAT_WS('-',SYMBOL,NAME) from CRYPTO_TOP_CURRENCIES WHERE SYMBOL='"+symbol+"'"
                    cursor.execute(SQL_select_data)
                    symbol_desc_rows = cursor.fetchall()                    
                    for row in symbol_desc_rows:
                        symbol_desc = row[0]
                    crypto_data = crypto_data_df.query("Crypto_Symbol == '"+symbol+"'")
                if (option == option4):
                    #call fbprophet model
                    model.buildFBProphetModel(crypto_data,symbol,symbol_desc)
                if (option == option5):                    
                    model.buildLSTMModel(crypto_data,symbol,symbol_desc)
                
            else:
                print("Invalid option!")

        finally:
            con.close()
    
    if __name__ == '__main__':
       performCryptoAnalysisTasks()
    
